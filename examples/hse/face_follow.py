import asyncio
import signal
from functools import partial
import cv2
import numpy as np
import logging
import time
import yaml
from os import path
from numpy import atan2, pi

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created


from go2_webrtc_driver.constants import VUI_COLOR
from go2_webrtc_driver.webrtc_driver import WebRTCConnectionMethod
from dog import Dog, ControlMode
from face_classifier import FaceClassifier



# Constants 
DEG2RAD = pi/180.0
RAD2DEG = 180.0/pi

MARKER_ID = 0
GO2_IP_ADDRESS = "192.168.4.204"
CAMERA_CALIBRATION_DATA = "ost.yaml"
V_MAX = 1.0
V_MIN = 0.25
W_MAX = 0.5
DIST_MIN = 0.4
DIST_FOLLOW = 1.0
DIST_ACC_MAX = 3.5
PHI_MAX = 0.2618

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
    camera_matrix = np.eye(3, dtype=np.float32)
    dist_coeffs = np.ones(5, dtype=np.float32)

    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
            camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
            dist_coeffs = np.array(data['distortion_coefficients']['data'])
    except FileNotFoundError:
        print("ERROR - File not found: " + yaml_file)
        print("Default camera parameters will be used.")

    return camera_matrix, dist_coeffs

async def tilt_to_pixel(dog: Dog, px, py, e=20):

    print(f"Target Pixel: ({px}, {py})") 

    # Optical center
    cx = dog.camera_matrix[0, 2]
    cy = dog.camera_matrix[1, 2]

    ex = cx - px
    ey = cy - py

    print(f"ex: {ex}  ey: {ey}")

    p = abs(0.00025 * e)

    if ex > e: dog.yaw += p
    elif ex < -e: dog.yaw -= p

    if ey > e: dog.pitch -= p/2
    elif ey < -e: dog.pitch += p/2

    print(f"Target rpy: ({dog.roll}, {dog.pitch}, {dog.yaw})")

    await dog.pose_rpy()

async def async_main():
    print("Hello from main")
    dog = Dog(WebRTCConnectionMethod.LocalSTA, ip_address=GO2_IP_ADDRESS)

    # Read camera parameters from YAML file
    camera_matrix, dist_coeffs = load_camera_parameters(CAMERA_CALIBRATION_DATA)
    dog.set_camera_parameters(camera_matrix, dist_coeffs)

    # Create an FaceClassifier object
    fc = FaceClassifier()

    stop_event = dog.stop_event

    # Create a new event loop for the asyncio code
    loop = asyncio.get_running_loop()

    # Assign own handler function to SIGINT signal
    signal.signal(signal.SIGINT, lambda s, f: handle_sigint(loop, dog))

    try:    
        await dog.startup_event()

        while not stop_event.is_set():
            if not dog.frame_queue.empty():
                img = dog.frame_queue.get()
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Use the VUI to indicate the current robot state
                # asyncio.create_task(dog.set_vui(VUI_COLOR.GREEN))

                c = 255
                color = (c, c, c)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.85
                thickness = 2

                predictions = fc.get_predictions(img)
                img = fc.draw_bounding_box(img, predictions)

                #text = f"some dummy text"
                #cv2.putText(img, text, (10, img.shape[0] - 11), font, scale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Video', img)
                key_input = cv2.waitKey(1)

                
                if key_input == 9: # Tab-Key
                    dog.toggle_mode()
                elif dog.mode is ControlMode.MODE_MANUAL.value:
                    dog.process_key(key_input, loop)
                elif dog.mode is ControlMode.MODE_AUTO.value:
                    # set pixel of interest to center of image as default
                    fx = dog.camera_matrix[0, 2]
                    fy = dog.camera_matrix[1, 2]

                    if len(predictions) > 0:
                        for p in predictions:
                            if p.get("name") == "Marco Dittmann" and p.get("probability") == 1.0:
                                bb = p.get("bbox")
                                fx = bb[0] + (bb[2] - bb[0]) / 2
                                fy = bb[1] + (bb[3] - bb[1]) / 2

                    asyncio.create_task(tilt_to_pixel(dog, fx, fy))

            else:
                await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(async_main())