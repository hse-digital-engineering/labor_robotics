import asyncio
import logging
import time
from enum import Enum
from queue import Queue
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack


class ControlMode(Enum):
    MODE_AUTO = "MODE_AUTO"
    MODE_MANUAL = "MODE_MANUAL"

class BatteryManagementSystem():
    def __init__(self, conn):
        self.conn = conn
        self.soc = 0.0 # State of Charge 
        self.current = 0 # Current draw (mA; negative number means discharging)

    def subscribe(self):
        self.conn.datachannel.pub_sub.subscribe(RTC_TOPIC['LOW_STATE'], self.lowstate_callback)

    def lowstate_callback(self, message):
        self.soc = message['data']['bms_state']['soc']
        self.current = message['data']['bms_state']['current']

class Dog:
    def __init__(self, connection_method: WebRTCConnectionMethod=WebRTCConnectionMethod.LocalSTA, ip_address: str="192.160.0.195"):
        self.conn = Go2WebRTCConnection(connection_method, ip=ip_address)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.vmax = 1.0 # translational velocity 
        self.wmax = 0.8 # rotational velocity
        self.bms = BatteryManagementSystem(self.conn)
        self.mode = "MODE_MANUAL"
        self.lidar_active = True
        self.marker_detected = False
        self.search_active = False
        self.last_detection_timestamp = 0
        self.camera_matrix, self.dist_coeffs = None, None
        self.frame_queue = Queue()
        self.stop_event = asyncio.Event() 
        self.task = None

    async def startup_event(self):
        self.task = asyncio.create_task(self.setup())

    async def shutdown_event(self):      
        if self.conn:
            await self.conn.disconnect()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("\n✅  Graceful shutdown after KeyboardInterrupt.")
        self.stop_event.set()
        

        
    def set_connection(self, ip_address: str, connection_method: WebRTCConnectionMethod):
        self.conn.ip = ip_address
        self.conn.connectionMethod = connection_method
    
    def connect(self):
        self.conn.connect()

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(self, track: MediaStreamTrack):
        while True: 
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            self.frame_queue.put(img)
        
    async def setup(self):
        try:
            while not self.stop_event.is_set():
                try:
                    await self.conn.connect()
                except SystemExit as e:
                    # logging.error(f"Connect exited with code {e.code}")
                    await asyncio.sleep(2)
                    continue  # try again
                except (asyncio.CancelledError, KeyboardInterrupt):
                    pass

                connected = self.check_connection()

                if connected:
                    print("✅ connected")
                    break

            self.conn.video.switchVideoChannel(True)
            self.conn.video.add_track_callback(self.recv_camera_stream)

            while self.bms is None:
                if self.stop_event.is_set():
                    print("Setup canceled by shutdown.")
                    return
                await asyncio.sleep(0.1)
                print("sleepy")

            self.bms.subscribe()

            await self.lidar_off()
            logging.info("Performing 'StandUp' movement...")
            await self.balance_stand()

        except Exception as e:
            logging.error(f"Error in WebRTC connection: {e}")


    def get_ip_address(self):
        return self.conn.ip

    def set_camera_parameters(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_vmax(self, v: float):
        self.v_max = v

    def set_wmax(self, w: float):
        self.w_max = w

    def set_mode(self, mode: ControlMode):
        if isinstance(mode, ControlMode):
            self.mode = mode
        else: 
            modes = [m.value for m in ControlMode]
            logging.warning("Invalid mode: " + mode + "\n" + \
                         f"Must be one of: {', '.join(modes)}")
            
    def toggle_mode(self):
        modes = [m.value for m in ControlMode] # get list of available modes
        logging.warning("modes: " + str(modes))
        idx = modes.index(self.mode) # current index
        idx = (idx + 1) % len(modes) # next index
        self.mode = modes[idx] # switch to next mode

    def get_soc(self):
        return self.bms.soc
    
    def get_current(self):
        return self.bms.current

    def process_key(self, key, loop):
        if key == ord('w'):
            # Move forward
            self.set_velocity(self.vmax, 0.0, 0.0)
        elif key == ord('a'):
            # Rotate counter-clockwise
            self.set_velocity(0.0, 0.0, self.wmax)  
        elif key == ord('s'):
            # Move backward
            self.set_velocity(-self.vmax, 0.0, 0.0)  
        elif key == ord('d'):
            # Rotate clockwise
            self.set_velocity(0.0, 0.0, -self.wmax)  
        elif key == ord('q'):
            # Move right
            self.set_velocity(0.0, 0.5, 0.0)
        elif key == ord('e'):
            # Move left
            self.set_velocity(0.0, -0.5, 0.0)
        elif key == ord('l'):
            # Turn on / off Lidar
            asyncio.run_coroutine_threadsafe(self.lidar_toggle(), loop)
        elif key == ord('1'):
            asyncio.run_coroutine_threadsafe(self.paw_wave(), loop)
        elif key == ord('2'):
            asyncio.run_coroutine_threadsafe(self.sit(), loop)
        elif key == ord('3'):
            asyncio.run_coroutine_threadsafe(self.stand_down(), loop)
        elif key == ord('4'):
            asyncio.run_coroutine_threadsafe(self.stand_up(), loop)
        elif key == ord('9'):
            asyncio.run_coroutine_threadsafe(self.jump_forward(), loop)
        else:
            # Stop movement
            self.set_velocity(0.0, 0.0, 0.0)
        asyncio.run_coroutine_threadsafe(self.move_xyz(), loop)
        asyncio.run_coroutine_threadsafe(self.pose_rpy(), loop)


    def set_velocity(self, vx: float, vy: float, vz: float):
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def set_rpy(self, roll: float, pitch: float, yaw: float):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


    async def set_vui(self, color: str):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return False

        colors = [value for key, value in VUI_COLOR.__dict__.items() if not key.startswith('__')]
        if color not in colors:
            logging.warning(f"Invalid color: {color}. Valid colors are: {colors}")
            return False
        else:
            logging.info("Setting VUI color...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["VUI"], {"api_id": 1007, 
                    "parameter": 
                    {
                        "color": color,
                    }
                }
            )
            return True

    def check_connection(self):
        if self.conn.isConnected:
            return True
        else:
            return False

    async def balance_stand(self):
        if self.check_connection():
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
                )

    async def jump_forward(self):
        if self.check_connection():
            logging.info("Jumping forward ...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["JumpForward"]}
            )

    async def paw_wave(self):
        if self.check_connection():
            logging.info("Performing 'Hello' movement ...")
            await self.conn.datachannel.pub_sub.publish_request_new(
               RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Hello"]}
            )


    async def stand_up(self):
        if self.check_connection():
            logging.info("Performing 'StandUp' movement ...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandUp"]}
            )

            logging.info("Performing 'StandUp' movement ...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["BalanceStand"]}
            )
        else:
            print("No connection")

    async def stand_down(self):
        if self.check_connection():
            logging.info("Performing 'StandDown' movement ...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["StandDown"]}
            )
    
    async def sit(self):
        if self.check_connection():
            logging.info("Performing 'Sit' movement...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"], {"api_id": SPORT_CMD["Sit"]}
            )

    async def pose_rpy(self):
        if self.check_connection():
            logging.info("Performing 'Euler' movement...")
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {"api_id": SPORT_CMD["Euler"],
                 "parameter": {"x": self.roll, "y": self.pitch, "z": self.yaw},
                },
            )

    async def move_xyz(self):
        if self.check_connection():
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

    
    async def lidar_off(self):
        await self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "OFF")
        self.lidar_active = False

    async def lidar_on(self):
        await self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "ON")
        self.lidar_active = True

    async def lidar_toggle(self):
        print("test")
        print(self.lidar_active)
        self.lidar_active = not self.lidar_active
        if not self.lidar_active:
            await self.lidar_on()
            print("lidar be turned on")
            
        else:
            await self.lidar_off()
            print("lidar will be turned off")
        

    async def find_marker(self, clockwise=False):
        w = -self.wmax if clockwise else self.wmax 
        if (time.time() - self.last_detection_timestamp) > 3.0 and self.search_active:
            self.set_velocity(0.0, 0.0, w)
        else:
            self.set_velocity(0.0, 0.0, 0.0)
        await self.move_xyz()