import asyncio
import signal
from functools import partial
import cv2
from cv2 import aruco
import numpy as np
import logging
import time
import yaml
from os import path
from numpy import atan2, pi
from go2_webrtc_driver.constants import VUI_COLOR
from go2_webrtc_driver.webrtc_driver import WebRTCConnectionMethod
from dog import Dog, ControlMode
from face_detection import FaceDetector

# Constants 
DEG2RAD = pi/180.0
RAD2DEG = 180.0/pi

MARKER_ID = 0
GO2_IP_ADDRESS = "192.168.4.201"
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

async def async_main():
    print("Hello from main")
    dog = Dog(WebRTCConnectionMethod.LocalSTA, ip_address=GO2_IP_ADDRESS)
    camera_matrix, dist_coeffs = load_camera_parameters(CAMERA_CALIBRATION_DATA)
    dog.set_camera_parameters(camera_matrix, dist_coeffs)

    fd = FaceDetector()

    stop_event = dog.stop_event
    loop = asyncio.get_running_loop()
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

                faces = fd.detect_faces(img)
                img = fd.draw_bounding_box(img, faces)

                #text = f"some dummy text"
                #cv2.putText(img, text, (10, img.shape[0] - 11), font, scale, color, thickness, cv2.LINE_AA)
                cv2.imshow('Video', img)
                key_input = cv2.waitKey(1)

                
                if key_input == 9: # Tab-Key
                    dog.toggle_mode()
                elif dog.mode == ControlMode.MODE_MANUAL:
                    dog.process_key(key_input, asyncio.get_event_loop())
                elif dog.mode == ControlMode.MODE_AUTO:
                    # tilt towards face
                    pass
            else:
                await asyncio.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(async_main())