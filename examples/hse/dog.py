import asyncio
import logging
import time
from enum import Enum
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod

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

class ControlMode(Enum):
    MODE_AUTO = "MODE_AUTO"
    MODE_MANUAL = "MODE_MANUAL"

class BatteryManagementSystem():
    def __init__(self, conn: Go2WebRTCConnection):
        self.conn = conn
        self.soc = 0.0   # State of Charge soc = 0.0
        self.current = 0 # Current draw (mA; negative number means discharging)current = 0.0

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
        self.vmax = 1.0 # translational velocity 
        self.wmax = 0.8 # rotational velocity
        self.bms = BatteryManagementSystem(self.conn)
        self.mode = "MODE_MANUAL"
        self.lidar_active = True
        self.marker_detected = False
        self.search_active = False
        self.last_detection_timestamp = 0
        self.camera_matrix, self.dist_coeffs = None, None
        
    def set_connection(self, ip_address: str, connection_method: WebRTCConnectionMethod):
        self.conn.ip = ip_address
        self.conn.connectionMethod = connection_method
    
    def connect(self):
        self.conn.connect()

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

    def check_connection(self):
        if not self.conn:
            logging.warning("Connection not established. Cannot perform movement.")
            return False
        else:
            return True

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