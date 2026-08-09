import time
import datetime
import asyncio
from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.offboard import Offboard, VelocityBodyYawspeed, OffboardError

class PX4Connector:
    def __init__(self, logger, system_address, on_arm_state_change=None):
        self.logger = logger
        self.system_address = system_address
        self.on_arm_state_change = on_arm_state_change
        self.drone = None
        self.is_armed = None

    def _format_ts(self, timestamp):
        """Converts epoch float to human-readable YYYY-MM-DD HH:MM:SS.fff"""
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    async def _monitor_arm_state(self):
        """Continuously monitors MAVLink telemetry for arm/disarm events."""
        try:
            async for is_armed in self.drone.telemetry.armed():
                # Silently capture the initial state so connect() can log the full summary block
                if self.is_armed is None:
                    self.is_armed = is_armed
                    
                # Log any subsequent changes to the arm state
                elif self.is_armed != is_armed:
                    state_str = "ARMED" if is_armed else "DISARMED"
                    self.logger.info(f"PX4 state changed to: {state_str}")
                    
                    if self.on_arm_state_change:
                        self.on_arm_state_change(is_armed)
                
                    self.is_armed = is_armed
        except Exception as e:
            self.logger.error(f"Failed to monitor arm state: {e}")

    async def connect(self):
        self.logger.info(f'Connecting to PX4 using MAVSDK on {self.system_address}')
        self.drone = System()
        await self.drone.connect(system_address=self.system_address)
        self.logger.info('Waiting for PX4 connection...')
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.logger.info('Connected to PX4!')
                break
                
        # Start monitoring the arm state in the background
        # Changed to ensure_future for Python 3.6 compatibility on VOXL2
        asyncio.ensure_future(self._monitor_arm_state())

        # Block the main thread until the first telemetry packet arrives
        self.logger.info('Waiting for initial telemetry stream...')
        while self.is_armed is None:
            await asyncio.sleep(0.1)
            
        # Fetch initial flight mode
        flight_mode = "UNKNOWN"
        async for mode in self.drone.telemetry.flight_mode():
            flight_mode = mode
            break
            
        # Fetch initial battery percentage
        battery_pct = 0.0
        async for battery in self.drone.telemetry.battery():
            battery_pct = battery.remaining_percent
            break

        # Output the drone's initial status block
        self.logger.info('====================================')
        self.logger.info('        DRONE INITIAL STATUS        ')
        self.logger.info('====================================')
        self.logger.info(f' Armed State : {"ARMED" if self.is_armed else "DISARMED"}')
        self.logger.info(f' Flight Mode : {flight_mode}')
        self.logger.info(f' Battery     : {int(battery_pct * 100)}%')
        self.logger.info('====================================')
        
        self.logger.info('Telemetry connection established. Starting detection processors...')

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
            
            # Send a dummy setpoint first to satisfy MAVSDK's internal requirement
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
            
            # Request offboard mode start
            await self.drone.offboard.start()
            self.logger.warning('PX4 Offboard mode successfully started')
            
        except OffboardError as e:
            self.logger.error(f'Failed to start Offboard mode (OffboardError): {e}')
        except Exception as e:
            self.logger.error(f'Unexpected MAVSDK error during Offboard start: {e}')