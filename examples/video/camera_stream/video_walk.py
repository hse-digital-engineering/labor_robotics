"""
Robot Video Feed and Movement Control Script
Based on: https://github.com/legion1581/go2_webrtc_connect

Author: Marco Dittmann
Date: 4.12.24  

Description:  
This script connects to a robotic dog and provides real-time video feed visualization.
The robot's movement can be controlled via keyboard inputs, allowing the user to navigate and perform predefined actions.
It uses OpenCV to display the video feed and asyncio for asynchronous robot communication.

Commands:  
- Movement Controls  
  - w Move forward  
  - s: Move backward  
  - a: Rotate counterclockwise (CCW)  
  - d: Rotate clockwise (CW)  
  - q: Sidestep left  
  - e: Sidestep right  

- Action Controls  
  - 1: Paw wave  
  - 2: Sit  
  - 3: Stand down  
  - 4: Stand up  

Features:  
- Real-time video feed display from the robot's camera.  
- Keyboard-controlled robot movement and actions.  
- Asynchronous WebRTC communication for controlling robot operations.  

Requirements:  
- Correctly set the IP address of the robot in the IP_ADDRESS constant.  
- Install required dependencies: OpenCV, asyncio, aiortc, and go2_webrtc_driver.  

Note:  
- Ensure the robot is connected and ready for remote operation before running the script.  
- Use appropriate safety measures to prevent damage or collisions during operation.  
"""

import cv2
import numpy as np

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
import logging
import threading
import time
from queue import Queue

# imports for movement
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

IP_ADDRESS = "192.168.0.199"
V_MAX = 2.0     # Maximum translational velocity (m/s)
W_MAX = 0.8     # Maximum rotational velocity (rad/s)


class Dog:
    def __init__(self, ip_address=IP_ADDRESS):
        self.ip_address = ip_address
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.conn = None

    def set_velocity(self, vx: float, vy: float, vz: float):
        self.vx = vx
        self.vy = vy
        self.vz = vz


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


dog = Dog()


def main():
    global dog
    frame_queue = Queue()

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
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()
                
                # Display the frame
                cv2.imshow('Video', img)
                key_input = cv2.waitKey(1)

                if key_input == ord('w'):
                    # Move forward
                    dog.set_velocity(V_MAX, 0.0, 0.0)
                elif key_input == ord('a'):
                    # Rotate counter-clockwise
                    dog.set_velocity(0.0, 0.0, W_MAX)  
                elif key_input == ord('s'):
                    # Move backward
                    dog.set_velocity(-V_MAX, 0.0, 0.0)  
                elif key_input == ord('d'):
                    # Rotate clockwise
                    dog.set_velocity(0.0, 0.0, -W_MAX)  
                elif key_input == ord('q'):
                    # Move right
                    dog.set_velocity(0.0, 0.5, 0.0)
                elif key_input == ord('e'):
                    # Move left
                    dog.set_velocity(0.0, -0.5, 0.0)
                elif key_input == ord('1'):
                    asyncio.run_coroutine_threadsafe(dog.paw_wave(), loop)
                elif key_input == ord('2'):
                    asyncio.run_coroutine_threadsafe(dog.sit(), loop)
                elif key_input == ord('3'):
                    asyncio.run_coroutine_threadsafe(dog.stand_down(), loop)
                elif key_input == ord('4'):
                    asyncio.run_coroutine_threadsafe(dog.stand_up(), loop)
                else:
                    # Stop movement
                    dog.set_velocity(0.0, 0.0, 0.0)

                asyncio.run_coroutine_threadsafe(dog.move_xyz(), loop)

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
