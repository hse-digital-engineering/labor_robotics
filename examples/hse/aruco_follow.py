import asyncio
import signal
from functools import partial
import cv2
from cv2 import aruco
import numpy as np
from numpy import atan2, pi

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created


import logging
import time
import yaml
from os import path
from numpy import atan2, pi
from go2_webrtc_driver.constants import VUI_COLOR
from go2_webrtc_driver.webrtc_driver import WebRTCConnectionMethod
from dog import Dog, ControlMode

# Constants 
DEG2RAD = pi/180.0
RAD2DEG = 180.0/pi

MARKER_ID = 0
GO2_IP_ADDRESS = "192.168.4.201"
CAMERA_CALIBRATION_DATA = "ost.yaml"
CENTER_ARUCO = True # Let the robot tilt its head to center the ArUco marker
V_MAX = 1.0         # Maximum translational velocity (m/s)
V_MIN = 0.25        # Maximum translational velocity (m/s)
W_MAX = 0.5         # Maximum rotational velocity (rad/s)
DIST_MIN = 0.4      # Distance and which robots starts to back off from ArUco
DIST_FOLLOW = 1.0   # Minimum distance at which the dog starts to follow the ArUco
DIST_ACC_MAX = 3.5  # Distance to ArUco where V_MAX is reached 
PHI_MAX = 0.2618    # Max angle at which the dog starts to center the ArUco

SHUTDOWN_IN_PROGRESS = False

def handle_sigint(loop, dog):
    global SHUTDOWN_IN_PROGRESS
    if SHUTDOWN_IN_PROGRESS:
        print("\r  \r⚠️  Shutdown already in progress. Please wait...")
        return
    SHUTDOWN_IN_PROGRESS = True
    print("\r  \r⌛️  Shutting down ...")

    loop.call_soon_threadsafe(asyncio.create_task, dog.shutdown_event())



def map(x, in_min, in_max, out_min, out_max):
    return out_min + (x - in_min)/(in_max - in_min) * (out_max - out_min)

def constrain(x, min_val, max_val):
    return min_val if x < min_val else (max_val if x > max_val else x)

def load_camera_parameters(yaml_file):
    # Default values in case the file does not exist
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.ones(5, dtype=np.float32)

    try:    
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    
            # Extract camera matrix
            camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
            
            # Extract distortion coefficients
            dist_coeffs = np.array(data['distortion_coefficients']['data'])
    except FileNotFoundError:
        print("ERROR - File not found: " + yaml_file)
        print("Default camera parameters will be used.")

    return camera_matrix, dist_coeffs

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

async def async_main():
    print("Hello from main")
    dog = Dog(WebRTCConnectionMethod.LocalSTA, ip_address=GO2_IP_ADDRESS)

    # Read camera parameters from YAML file
    camera_matrix, dist_coeffs = load_camera_parameters(CAMERA_CALIBRATION_DATA)
    dog.set_camera_parameters(camera_matrix, dist_coeffs)

    # ArUco parameter setup
    aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dictionary, detector_parameters)
    marker_size = 0.15 # 150 mm

    stop_event = dog.stop_event

    # Create a new event loop for the asyncio code
    loop = asyncio.get_running_loop()

    # Assign own handler function to SIGINT signal
    signal.signal(signal.SIGINT, lambda s, f: handle_sigint(loop, dog))

    try:    
        await dog.startup_event()

        aruco_x = aruco_y = aruco_z = None
        phi, chi = None
        text = "Position: unknown"
        text_phi = "phi: unknown"
        cx, cy = img/2, img/2 # Center of the image

        while not stop_event.is_set():
            if not dog.frame_queue.empty():
                img = dog.frame_queue.get()
                img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                corners, ids, rejected = detector.detectMarkers(img_greyscale)
                if ids is not None:
                    aruco.drawDetectedMarkers(img, corners, ids)
                    if any(id == 0 for id in ids):
                        dog.marker_detected = True
                        dog.search_active = False
                        dog.last_detection_timestamp = time.time()
                        asyncio.create_task(dog.set_vui(VUI_COLOR.GREEN))

                        # Perform pose estimation here
                        rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_size, dog.camera_matrix, dog.dist_coeffs)
                        aruco_x, aruco_y, aruco_z = tvecs[0][0][0], tvecs[0][1][0], tvecs[0][2][0]
                        phi = atan2(aruco_x, aruco_z) # horizontal angle to ArUco in front of camera
                        chi = atan2(aruco_y, aruco_z) # vertical angle to ArUco in front of camera
                    else:
                        dog.marker_detected = False
                        asyncio.create_task(dog.set_vui(VUI_COLOR.BLUE))

                # Adaptive text color
                brightness = np.mean(img_greyscale[-50, :]) # 0 - 255
                c = int(255 - brightness)
                color = (c, c, c)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.85
                thickness = 2

                # Write pose data as text into the image
                if dog.marker_detected and aruco_z is not None:
                    text = f"Position: ({aruco_x:.3f}, {aruco_y:.3f}, {aruco_z:.3f})"
                    text_phi =  f"phi: {phi * RAD2DEG:.1f} deg"
                else:
                    text = "Position: unknown"
                    text_phi = "phi: unknown"

                cv2.putText(img, text, (10, img.shape[0] - 11), font, scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img, "Mode: " + dog.mode[len("MODE_" ):], (10, img.shape[0] - 51), font, scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img, text_phi, (img.shape[1] - 251, img.shape[0] - 11), font, scale, color, thickness, cv2.LINE_AA)
                cv2.putText(img, f"Battery: {dog.get_soc()} %", (img.shape[1] - 271, 31), font, scale, (255,255,255), thickness, cv2.LINE_AA)
                cv2.putText(img, f"Current: {dog.get_current()} mA", (img.shape[1] - 271, 71), font, scale, (255,255,255), thickness, cv2.LINE_AA)

                # Display the current video frame 
                cv2.imshow('Video', img)
                key_input = cv2.waitKey(1)

                if key_input == 9: # Tab-Key
                    dog.toggle_mode()
                elif dog.mode is ControlMode.MODE_MANUAL.value:
                    dog.process_key(key_input, loop)
                    dog.set_rpy(0., 0., 0.)
                elif dog.mode is ControlMode.MODE_AUTO.value:
                    if aruco_z:
                        # Filter estimations outside reasonable position constraints
                        x_valid = -5.0 < aruco_x < 5.0
                        y_valid = -1.0 < aruco_y < 1.0
                        z_valid = 0.0 < aruco_z < 10.0
                        if dog.marker_detected and all([x_valid, y_valid, z_valid]):

                            # Set pitch and yaw to center the ArUco marker in the image
                            if CENTER_ARUCO:
                                if abs(aruco_x - cx) > 10:
                                    dog.yaw = phi
                                if abs(aruco_y - cy) > 10:
                                    dog.pitch = chi

                            if aruco_z > DIST_FOLLOW:
                                # Increase translational velocity gradually depending on current distance
                                # V_MAX at distance of DIST_ACC_MAX
                                vx = map(aruco_z, DIST_FOLLOW, DIST_ACC_MAX, V_MIN, V_MAX)

                                # Keep velocity within boundaries
                                vx = constrain(vx, V_MIN, V_MAX)
                                dog.vx, dog.vy, dog.vz = vx, 0.0, -aruco_x/aruco_z
                            elif aruco_z < DIST_MIN:
                                dog.vx, dog.vy, dog.vz = -V_MIN, 0.0, 0.0
                            else:
                                vz = -W_MAX if phi > PHI_MAX else (W_MAX if -phi > PHI_MAX else 0.0)
                                dog.vx, dog.vy, dog.vz = 0.0, 0.0, vz
                            asyncio.create_task(dog.move_xyz())
                        else:
                            dog.search_active = True
                            dog.set_rpy(0.0, 0.0, 0.0)
                            # In which direction did the ArUco marker disappear?
                            if aruco_x is None or aruco_x < 0:
                                asyncio.create_task(dog.find_marker(clockwise=False))
                            else:
                                asyncio.create_task(dog.find_marker(clockwise=True))
            else:
                await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(async_main())