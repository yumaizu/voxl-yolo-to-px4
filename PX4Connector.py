import time
import datetime
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import Offboard, OffboardError

class PX4Connector:
    def __init__(self, logger, system_address):
        self.logger = logger
        self.system_address = system_address
        self.drone = None

    def _format_ts(self, timestamp):
        """Converts epoch float to human-readable YYYY-MM-DD HH:MM:SS.fff"""
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    async def connect(self):
        self.logger.info(f'Connecting to PX4 using MAVSDK on {self.system_address}')
        self.drone = System()
        await self.drone.connect(system_address=self.system_address)
        self.logger.info('Waiting for PX4 connection...')
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.logger.info('Connected to PX4!')
                break

    async def set_hold_mode(self):
        try:
            mavlink_timestamp = time.time()
            formatted_time = self._format_ts(mavlink_timestamp)
            
            self.logger.warning(f'[{formatted_time}] PERSON DETECTED: Sending Hold mode signal to PX4')
            await self.drone.action.hold()
            self.logger.warning('PX4 Hold command successfully sent')
            
        except ActionError as e:
            self.logger.error(f'Failed to send Hold command: {e}')
        except Exception as e:
            self.logger.error(f'Unexpected MAVSDK error: {e}')

    async def set_land_mode(self):
        try:
            mavlink_timestamp = time.time()
            formatted_time = self._format_ts(mavlink_timestamp)
            
            self.logger.warning(f'[{formatted_time}] PERSON DETECTED: Sending Land mode signal to PX4')
            await self.drone.action.land()
            self.logger.warning('PX4 Land command successfully sent')
            
        except ActionError as e:
            self.logger.error(f'Failed to send Land command: {e}')
        except Exception as e:
            self.logger.error(f'Unexpected MAVSDK error: {e}')

    async def set_offboard_mode(self):
        try:
            mavlink_timestamp = time.time()
            formatted_time = self._format_ts(mavlink_timestamp)
            
            self.logger.warning(f'[{formatted_time}] PERSON DETECTED: Requesting Offboard mode switch')
            
            # Since an external process is already streaming setpoints, 
            # we only need to tell MAVSDK to start/activate offboard mode.
            await self.drone.offboard.start()
            self.logger.warning('PX4 Offboard mode successfully started')
            
        except OffboardError as e:
            self.logger.error(f'Failed to start Offboard mode (OffboardError): {e}')
        except Exception as e:
            self.logger.error(f'Unexpected MAVSDK error during Offboard start: {e}')