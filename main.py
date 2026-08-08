#!/usr/bin/env python3

import asyncio
import threading
import logging
import configparser
import sys

from YOLOProcessor import YOLOProcessor
from PX4Connector import PX4Connector
from VisionHubConnector import VisionHubConnector

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

config = configparser.ConfigParser()
if not config.read('config.ini'):
    logger.warning('Failed to read config.ini, using default config.ini')
    config.read('config_default.ini')

def main():
    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(
        target=loop.run_forever,
        daemon=True
    )
    asyncio_thread.start()

    # Determine which connector to use from the GENERAL section
    CONNECTOR_TYPE = config.get('GENERAL', 'CONNECTOR_TYPE', fallback='VISION_HUB').upper()
    logger.info(f'Using connector type: {CONNECTOR_TYPE}')

    actions = {}

    if CONNECTOR_TYPE == 'PX4':
        connector = PX4Connector(
            logger=logger,
            system_address=config.get('PX4', 'ADDRESS')
        )
        actions = {
            'hold': connector.set_hold_mode,
            'land': connector.set_land_mode,
            'offboard': connector.set_offboard_mode
        }
        
        # Connect asynchronously 
        future = asyncio.run_coroutine_threadsafe(
            connector.connect(),
            loop
        )
        try:
            future.result() # Wait for the connection to establish
        except Exception as e:
            logger.error(f'Failed to connect to PX4: {e}')
            loop.call_soon_threadsafe(loop.stop)
            asyncio_thread.join()
            sys.exit(1)

    elif CONNECTOR_TYPE == 'VISION_HUB':
        # Retrieve network config from the VISIONHUB section
        drone_ip = config.get('VISIONHUB', 'DRONE_IP', fallback='192.168.8.1')
        trigger_port = config.getint('VISIONHUB', 'TRIGGER_PORT', fallback=5005)

        connector = VisionHubConnector(
            logger=logger,
            drone_ip=drone_ip,
            port=trigger_port
        )

        # Wrap the synchronous UDP send in an async function for the YOLO loop
        async def async_trigger_descent():
            connector.trigger_descent()

        actions = {
            'hold': async_trigger_descent,
            'land': async_trigger_descent,
            'offboard': async_trigger_descent
        }
        
        # Connect synchronously
        connector.connect()

    else:
        logger.error(f'Unknown CONNECTOR_TYPE: {CONNECTOR_TYPE}. Must be "PX4" or "VISION_HUB".')
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()
        sys.exit(1)


    # Validate the chosen detection action
    DETECTION_ACTION = config.get('PX4', 'DETECTION_ACTION', fallback='offboard')
    if DETECTION_ACTION not in actions:
        logger.error(f'Unknown detection action: {DETECTION_ACTION}')
        logger.error(f'Available actions: {", ".join(actions.keys())}')
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()
        return

    logger.info(f'Detection action configured: {DETECTION_ACTION}')

    yolo_processor = None
    web_view = None
    
    try:
        enable_web_stream = config.getboolean('YOLO', 'ENABLE_WEB_STREAM', fallback=True)
        
        # Instantiate and start the web server if enabled
        if enable_web_stream:
            from YOLOWebView import YOLOWebView
            web_port = config.get('YOLO', 'WEB_PORT', fallback='8080')
            web_view = YOLOWebView(port=web_port, logger=logger)
            web_view.start()

        yolo_processor = YOLOProcessor(
            loop=loop,
            rtsp_url=config.get('YOLO', 'RTSP_URL'),
            model_path=config.get('YOLO', 'MODEL_PATH'),
            confidence_threshold=config.get('YOLO', 'CONFIDENCE_THRESHOLD'),
            device=config.get('YOLO', 'DEVICE', fallback='auto'),
            enable_debug_window=config.getboolean('YOLO', 'ENABLE_DEBUG_WINDOW', fallback=False),
            web_view=web_view,
            artificial_latency_ms=config.get('GENERAL', 'ARTIFICIAL_LATENCY_MS', fallback='0'),
            action_callback=actions[DETECTION_ACTION],
            logger=logger
        )

        logger.info('YOLO Processor is ready')
        yolo_processor.start()

    except KeyboardInterrupt:
        logger.info('Shutting down')
    finally:
        if yolo_processor:
            yolo_processor.stop()
        if web_view:
            web_view.stop()
        
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == '__main__':
    main()