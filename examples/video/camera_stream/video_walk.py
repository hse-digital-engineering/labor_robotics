"""
Dog Control Program with WebRTC and OpenCV Integration

This program connects to a robotic dog using a WebRTC-based connection to 
control its movement and display a video stream from its camera. The user can 
control the dog using keyboard inputs (`w`, `a`, `s`, `d` for movement and ESC to exit). 

Author: Marco Dittmann
Date: 29.11.2024
"""

import asyncio
import cv2
import logging
import threading
import json
from time import sleep
from queue import Queue

from aiortc import MediaStreamTrack
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

IP_ADDRESS = "192.168.0.1"
logging.basicConfig(level=logging.ERROR)


class Dog:
    def __init__(self, ip_address=IP_ADDRESS):
        self.ip_address = ip_address
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.conn = None

    async def paw_wave(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform paw wave.")
            return

        logging.info("Performing 'Hello' movement...")
        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Hello"]}
        )

    async def move_xyz(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot move.")
            return

        await self.conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            {
                "api_id": SPORT_CMD["Move"],
                "parameter": {"x": self.vx, "y": self.vy, "z": self.vz},
            },
        )


async def recv_camera_stream(track: MediaStreamTrack, frame_queue: Queue):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC-Key
            break


def run_asyncio_loop(loop: asyncio.AbstractEventLoop, dog: Dog):
    asyncio.set_event_loop(loop)

    async def setup():
        try:
            await dog.conn.connect()
            logging.info("Connection established.")

            response = await dog.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
            )

            if response["data"]["header"]["status"]["code"] == 0:
                data = json.loads(response["data"]["data"])
                current_mode = data.get("name", "unknown")
                logging.info(f"Current motion mode: {current_mode}")

            if current_mode != "normal":
                logging.info("Switching to 'normal' mode...")
                await dog.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["MOTION_SWITCHER"],
                    {"api_id": 1002, "parameter": {"name": "normal"}},
                )
                await asyncio.sleep(5)
                await dog.paw_wave()

            dog.conn.video.switchVideoChannel(True)
            dog.conn.video.add_track_callback(lambda track: recv_camera_stream(track, Queue()))

        except Exception as e:
            logging.error(f"Error during setup: {e}")

    loop.run_until_complete(setup())
    loop.run_forever()


def main():
    frame_queue = Queue()
    dog = Dog()

    dog.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=dog.ip_address)

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop, dog))
    asyncio_thread.start()

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC-Key
                print("The program was terminated by the user.")
                break
            elif key == ord("w"):
                dog.vx, dog.vy = 0.5, 0.0
            elif key == ord("a"):
                dog.vx, dog.vy = 0.0, -0.5
            elif key == ord("s"):
                dog.vx, dog.vy = -0.5, 0.0
            elif key == ord("d"):
                dog.vx, dog.vy = 0.0, 0.5
            else:
                dog.vx, dog.vy = 0.0, 0.0
                sleep(0.01)

            asyncio.run(dog.move_xyz())
    except KeyboardInterrupt:
        print("The program was terminated by Ctrl+C.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()


if __name__ == "__main__":
    main()
