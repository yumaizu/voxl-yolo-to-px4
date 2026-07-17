from mavsdk import System
from mavsdk.action import ActionError


class PX4Connector:

    def __init__(self, logger, system_address):

        self.logger = logger
        self.system_address = system_address
        self.drone = None

    async def connect(self):

        self.logger.info(
            'Connecting to PX4 using MAVSDK on {}'.format(
                self.system_address
            )
        )

        self.drone = System()

        await self.drone.connect(system_address=self.system_address)

        self.logger.info('Waiting for PX4 connection...')

        async for state in self.drone.core.connection_state():

            if state.is_connected:

                self.logger.info('Connected to PX4!')

                break

    async def set_hold_mode(self):
        try:
            self.logger.warn('Sending Hold mode command to PX4')

            await self.drone.action.hold()

            self.logger.warn('PX4 Hold command successfully sent')

        except ActionError as e:
            self.logger.error('Failed to send Hold command: {}'.format(e))

        except Exception as e:
            self.logger.error('Unexpected MAVSDK error: {}'.format(e))

    async def set_land_mode(self):
        try:
            self.logger.warn('Sending Land command to PX4')

            await self.drone.action.land()

            self.logger.warn('PX4 Land command successfully sent')

        except ActionError as e:
            self.logger.error('Failed to send Land command: {}'.format(e))

        except Exception as e:
            self.logger.error('Unexpected MAVSDK error: {}'.format(e))