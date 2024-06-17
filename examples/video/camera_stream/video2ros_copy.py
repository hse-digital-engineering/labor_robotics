"""
Robot Control Script for WebRTC and Marker Detection
Based on: https://github.com/legion1581/go2_webrtc_connect

Author: Marco Dittmann  
Date: 11.12.24  

Description:  
This script turns on the robot's camera and publishes the image as a ROS message of type sensor_msgs/Image.

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

# ROS 2 Imports
import rclpy
from rclpy.node import Node
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

# WebRTC and Asyncio Imports
import asyncio
import logging
import threading
import time
from queue import Queue
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

IP_ADDRESS = "192.168.0.199"

# class ImagePublisher(Node):
#     def __init__(self):
#         super().__init__('image_publisher')

#         # Create a publisher for the ROS Image message
#         self.publisher_ = self.create_publisher(Image, 'image_topic', 10)

#         # Initialize a CvBridge to convert OpenCV images to ROS Image messages
#         self.bridge = CvBridge()

#     def publish_image(self, cv_image):
#         try:
#             ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
#             ros_image.header.stamp = self.get_clock().now().to_msg()
#             self.publisher_.publish(ros_image)
#             self.get_logger().info('Image published!')
#         except Exception as e:
#             self.get_logger().error(f'Failed to publish image: {e}')

def main():
    # rclpy.init()
    frame_queue = Queue()

    # ROS Node
    # image_publisher = ImagePublisher()
    # image_publisher.get_logger().info("hello from ROS")

    # Choose a connection method (uncomment the correct one)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=IP_ADDRESS)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)


            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

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
                print(f"Shape: {img.shape}, Dimensions: {img.ndim}, Type: {img.dtype}, Size: {img.size}")
                # Display the frame
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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
