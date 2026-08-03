#!/usr/bin/env python3

import asyncio
import threading

import rclpy

from YOLOReceiver import YOLOReceiver
from PX4Connector import PX4Connector

import configparser

config = configparser.ConfigParser()
if not config.read('config.ini'):
    print('Failed to read config.ini, using default config.ini')
    config.read('config_default.ini')

def main():

    loop = asyncio.new_event_loop()

    asyncio_thread = threading.Thread(
        target=loop.run_forever,
        daemon=True
    )

    asyncio_thread.start()

    rclpy.init()

    yolo_receiver = YOLOReceiver(
        loop=loop,
        detection_topic=config.get('YOLO', 'DETECTION_TOPIC'),
        action_callback=None,
        pipe_prefix=config.get('YOLO', 'TFLITE_OUTPUT_PIPE_PREFIX')
    )

    px4_connector = PX4Connector(
        logger=yolo_receiver.get_logger(),
        system_address=config.get('PX4', 'ADDRESS')
    )

    actions = {

        'hold': px4_connector.set_hold_mode,

        'land': px4_connector.set_land_mode,

    }

    DETECTION_ACTION = config.get('YOLO', 'DETECTION_ACTION')
    if DETECTION_ACTION not in actions:

        yolo_receiver.get_logger().error(
            'Unknown detection action: {}'.format(
                DETECTION_ACTION
            )
        )

        yolo_receiver.get_logger().error(
            'Available actions: {}'.format(
                ', '.join(actions.keys())
            )
        )

        yolo_receiver.destroy_node()

        rclpy.shutdown()

        loop.call_soon_threadsafe(
            loop.stop
        )

        asyncio_thread.join()

        return

    yolo_receiver.action_callback = (
        actions[DETECTION_ACTION]
    )

    yolo_receiver.get_logger().info(
        'Detection action configured: {}'.format(
            DETECTION_ACTION
        )
    )

    future = asyncio.run_coroutine_threadsafe(
        px4_connector.connect(),
        loop
    )

    try:

        # Wait for MAVSDK connection
        future.result()

        yolo_receiver.get_logger().info(
            'YOLO receiver is ready'
        )

        # Start ROS 2 event loop
        rclpy.spin(yolo_receiver)

    except KeyboardInterrupt:

        yolo_receiver.get_logger().info(
            'Shutting down'
        )

    finally:

        yolo_receiver.destroy_node()

        rclpy.shutdown()

        loop.call_soon_threadsafe(
            loop.stop
        )

        asyncio_thread.join()


if __name__ == '__main__':

    main()