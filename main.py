#!/usr/bin/env python3

import asyncio
import threading
import logging
import configparser
import sys
import time

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

    # -------------------------------------------------
    # Connector Selection (PX4 vs VISION_HUB)
    # -------------------------------------------------
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
        
        future = asyncio.run_coroutine_threadsafe(
            connector.connect(),
            loop
        )
        try:
            future.result()
        except Exception as e:
            logger.error(f'Failed to connect to PX4: {e}')
            loop.call_soon_threadsafe(loop.stop)
            asyncio_thread.join()
            sys.exit(1)

    elif CONNECTOR_TYPE == 'VISION_HUB':
        drone_ip = config.get('VISIONHUB', 'DRONE_IP', fallback='127.0.0.1')
        trigger_port = config.getint('VISIONHUB', 'TRIGGER_PORT', fallback=5005)

        connector = VisionHubConnector(
            logger=logger,
            drone_ip=drone_ip,
            port=trigger_port
        )

        async def async_trigger_descent():
            connector.trigger_descent()

        actions = {
            'hold': async_trigger_descent,
            'land': async_trigger_descent,
            'offboard': async_trigger_descent
        }
        
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

    # -------------------------------------------------
    # YOLO Mode Selection (python vs voxl-tflite-server)
    # -------------------------------------------------
    YOLO_MODE = config.get('GENERAL', 'YOLO_MODE', fallback='python').strip().lower()
    logger.info(f'Using YOLO mode: {YOLO_MODE}')

    yolo_processor = None
    web_view = None
    yolo_receiver = None
    rclpy_module = None

    try:
        if YOLO_MODE == 'python':
            # Dynamically import YOLOProcessor to prevent cv2/ultralytics crashes on the drone
            try:
                from YOLOProcessor import YOLOProcessor
            except ImportError as e:
                logger.error(f'Failed to import YOLO modules (cv2/ultralytics): {e}')
                sys.exit(1)

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

            # Keep main thread active for PyTorch/OpenCV execution
            while True:
                time.sleep(1)

        elif YOLO_MODE == 'voxl-tflite-server':
            # Dynamic import to prevent dependency errors on non-ROS systems
            try:
                import rclpy
                from YOLOReceiver import YOLOReceiver
                rclpy_module = rclpy
            except ImportError as e:
                logger.error(f'Failed to import ROS 2 / rclpy modules for voxl-tflite-server mode: {e}')
                sys.exit(1)

            rclpy_module.init()

            detection_topic = config.get('YOLO', 'DETECTION_TOPIC', fallback='/tflite/aidetection')
            pipe_prefix = config.get('YOLO', 'PIPE_PREFIX', fallback='')

            yolo_receiver = YOLOReceiver(
                loop=loop,
                detection_topic=detection_topic,
                action_callback=actions[DETECTION_ACTION],
                pipe_prefix=pipe_prefix
            )

            logger.info('YOLO Receiver is ready. Spinning ROS 2 node...')
            rclpy_module.spin(yolo_receiver)

        else:
            logger.error(f'Unknown YOLO_MODE: {YOLO_MODE}. Must be "python" or "voxl-tflite-server".')

    except KeyboardInterrupt:
        logger.info('Shutting down via KeyboardInterrupt')
    finally:
        if yolo_processor:
            yolo_processor.stop()
        if web_view:
            web_view.stop()
        if yolo_receiver:
            yolo_receiver.destroy_node()
        if rclpy_module and rclpy_module.ok():
            rclpy_module.shutdown()
        
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == '__main__':
    main()