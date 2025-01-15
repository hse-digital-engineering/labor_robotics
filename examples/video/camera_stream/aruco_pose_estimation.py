"""
Robot Control Script for WebRTC and Marker Detection
Based on: https://github.com/legion1581/go2_webrtc_connect

Author: Marco Dittmann  
Date: 4.12.24  

Description:  
This script uses the Dog class to control the movement of a robotic dog and utilizes its camera feed to search for and detect ArUco markers.
The robot's behavior dynamically adapts based on marker detection, allowing it to perform various actions, such as moving, sitting, and standing.  
The script integrates OpenCV for video processing, asyncio for asynchronous communication, and WebRTC for establishing a connection with the robot.
It processes video frames in real-time to detect markers and triggers appropriate actions based on the detection state.  

Features:  
- Real-time video streaming and marker detection with OpenCV and ArUco.  
- Robot movement control (e.g., move, sit, stand) using the Dog class.  
- Asynchronous communication with the robot via WebRTC.  
- Integration with threading for smooth operation.  

Note:  
- Ensure the robot's IP address is correctly set in the IP_ADDRESS constant.  
- Install all dependencies, including OpenCV, asyncio, and aiortc.  
"""


import cv2
from cv2 import aruco
import numpy as np

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
from enum import Enum
import logging
import threading
import time
import yaml
from os import path
from datetime import datetime
from queue import Queue

## imports for movement
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack


# Enable logging for debugging
logging.basicConfig(level=logging.WARN)

IP_ADDRESS = "192.168.0.199"
CAMERA_CALIBRATION_DATA = "ost.yaml"
V_MAX = 0.8     # Maximum translational velocity (m/s)
W_MAX = 0.5     # Maximum rotational velocity (rad/s)


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
        print("The camera matrix will be set to the unity matrix.\n \
              The distortion coefficients will be set to a vector of ones.")

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

class ControlMode(Enum):
    MODE_AUTO = "MODE_AUTO"
    MODE_MANUAL = "MODE_MANUAL"

class Dog:
    def __init__(self, ip_address=IP_ADDRESS):
        self.ip_address = ip_address
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.conn = None
        self.mode = "MODE_AUTO"
        self.marker_detected = False
        self.search_active = False
        self.last_detection_timestamp = 0
        self.camera_matrix, self.dist_coeffs = load_camera_parameters(CAMERA_CALIBRATION_DATA)

    def set_mode(self, mode: ControlMode):
        if isinstance(mode, ControlMode):
            self.mode = mode
        else: 
            modes = [m.value for m in ControlMode]
            logging.warn("Invalid mode: " + mode + "\n" + \
                         f"Must be one of: {', '.join(modes)}")
            
    def toggle_mode(self):
        modes = [m.value for m in ControlMode] # get list of available modes
        logging.warn("modes: " + str(modes))
        idx = modes.index(self.mode) # current index
        idx = (idx + 1) % len(modes) # next index
        self.mode = modes[idx] # switch to next mode

    def process_key(self, key, loop):
        if key == ord('w'):
            # Move forward
            self.set_velocity(V_MAX, 0.0, 0.0)
        elif key == ord('a'):
            # Rotate counter-clockwise
            self.set_velocity(0.0, 0.0, W_MAX)  
        elif key == ord('s'):
            # Move backward
            self.set_velocity(-V_MAX, 0.0, 0.0)  
        elif key == ord('d'):
            # Rotate clockwise
            self.set_velocity(0.0, 0.0, -W_MAX)  
        elif key == ord('q'):
            # Move right
            self.set_velocity(0.0, 0.5, 0.0)
        elif key == ord('e'):
            # Move left
            self.set_velocity(0.0, -0.5, 0.0)
        elif key == ord('1'):
            asyncio.run_coroutine_threadsafe(self.paw_wave(), loop)
        elif key == ord('2'):
            asyncio.run_coroutine_threadsafe(self.sit(), loop)
        elif key == ord('3'):
            asyncio.run_coroutine_threadsafe(self.stand_down(), loop)
        elif key == ord('4'):
            asyncio.run_coroutine_threadsafe(self.stand_up(), loop)
        else:
            # Stop movement
            self.set_velocity(0.0, 0.0, 0.0)
        asyncio.run_coroutine_threadsafe(self.move_xyz(), loop)


    def set_velocity(self, vx: float, vy: float, vz: float):
        self.vx = vx
        self.vy = vy
        self.vz = vz


    async def set_vui(self, color):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return

        logging.info("Setting VUI color...")
        if True:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["VUI"], {"api_id": 1007, 
                    "parameter": 
                    {
                        "color": color,
                    }
                }
            )

    async def paw_wave(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return

        logging.info("Performing 'Hello' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Hello"]}
        )

    async def stand_up(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return

        logging.info("Performing 'StandUp' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
        )

        logging.info("Performing 'StandUp' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
        )

    async def stand_down(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return

        logging.info("Performing 'StandDown' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]}
        )
    
    async def sit(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return

        logging.info("Performing 'Sit' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Sit"]}
        )

    async def move_xyz(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot move.")
            return

        if self.vx == 0.0 and self.vy == 0.0 and self.vz == 0.0:
            pass
        else:
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": self.vx, "y": self.vy, "z": self.vz},
                },
            )
        logging.info(f"Moving robot: vx={self.vx}, vy={self.vy}, vz={self.vz}")

    async def find_marker(self):
        if (time.time() - self.last_detection_timestamp) > 3.0 and self.search_active:
            self.set_velocity(0.0, 0.0, W_MAX)
        else:
            self.set_velocity(0.0, 0.0, 0.0)
        await self.move_xyz()


dog = Dog(IP_ADDRESS)



def main():
    global dog
    frame_queue = Queue()

    aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector_parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dictionary, detector_parameters)
    marker_size = 0.15 # 150 mm

    # Choose a connection method (uncomment the correct one)
    dog.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=dog.ip_address)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                # Connect to the device
                await dog.conn.connect()

                # Switch video channel on and start receiving video frames
                dog.conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                dog.conn.video.add_track_callback(recv_camera_stream)

                logging.info("Performing 'StandUp' movement...")
                await dog.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
                )
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    try:
        find_marker = True
        corners = None
        rvecs = None
        tvecs = None
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                img_greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(img_greyscale)
                if ids is not None:
                    aruco.drawDetectedMarkers(img, corners, ids)
                    dog.marker_detected = True
                    dog.last_detection_timestamp = time.time()
                    asyncio.run_coroutine_threadsafe(dog.set_vui(VUI_COLOR.GREEN), loop)

                    # Perform pose estimation here
                    rvecs, tvecs, trash = my_estimatePoseSingleMarkers(corners, marker_size, dog.camera_matrix, dog.dist_coeffs)

                    # Write pose data as text into the image
                    text = f"Position: ({tvecs[0][0][0]:.3f}, {tvecs[0][1][0]:.3f}, {tvecs[0][2][0]:.3f})"
                    thickness = 2
                    scale = 0.85
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (255, 255, 255)
                    origin = (10, img.shape[0] - 11)
                    cv2.putText(img, text, origin, font, scale, color, thickness, cv2.LINE_AA)
                    text = "Mode: " + dog.mode[len("MODE_"):]
                    origin = (10, img.shape[0] - 51)
                    cv2.putText(img, text, origin, font, scale, color, thickness, cv2.LINE_AA)

                    # Move Dog towards marker, keep distance < 3.0 m
                    # asyncio.run_coroutine_threadsafe(dog.move_xyz(), loop)

                else:
                    dog.marker_detected = False
                    asyncio.run_coroutine_threadsafe(dog.set_vui(VUI_COLOR.BLUE), loop)
                    
                
                # Display the frame
                cv2.imshow('Video', img)
                key_input = cv2.waitKey(1)

                if key_input == 9: # Tab-Key
                    dog.toggle_mode()

                elif dog.mode is ControlMode.MODE_MANUAL.value:
                    dog.process_key(key_input, loop)


            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
