#!/usr/bin/env python3

import asyncio
import threading
import logging
import configparser

from YOLOProcessor import YOLOProcessor
from PX4Connector import PX4Connector

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

    px4_connector = PX4Connector(
        logger=logger,
        system_address=config.get('PX4', 'ADDRESS')
    )

    actions = {
        'hold': px4_connector.set_hold_mode,
        'land': px4_connector.set_land_mode,
    }

    DETECTION_ACTION = config.get('PX4', 'DETECTION_ACTION')
    if DETECTION_ACTION not in actions:
        logger.error(f'Unknown detection action: {DETECTION_ACTION}')
        logger.error(f'Available actions: {", ".join(actions.keys())}')
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()
        return

    logger.info(f'Detection action configured: {DETECTION_ACTION}')

    future = asyncio.run_coroutine_threadsafe(
        px4_connector.connect(),
        loop
    )

    yolo_processor = None
    web_view = None
    
    try:
        future.result()

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