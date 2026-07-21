#!/usr/bin/env python3

import asyncio
import threading

import rclpy

from YOLOReceiver import YOLOReceiver
from PX4Connector import PX4Connector


# Configuration
# =========================================================

"""
PX4_ADDRESS : str
The address of the remote system. If None, it will default to
udpin://0.0.0.0:14540.

Supported URL formats:

        - Serial: serial:///path/to/serial/dev[:baudrate]
        - UDP in: udpin://bind_host:bind_port
        - UDP out: udpout://dest_host:dest_port
        - TCP in: tcpin://bind_host:bind_port
        - TCP out: tcpout://dest_host:dest_port

for VOXL2 change the following lines in /etc/modalai/voxl-vision-hub.conf
        "en_localhost_mavlink_udp":     true,  // false -> true
        "localhost_udp_port_number":    14551, // set the same port bellow
"""
PX4_ADDRESS = 'udp://:14551'


"""
YOLO_DETECTION_TOPIC : str
The ROS 2 topic to listen for YOLO detections on.

Message type:
    voxl_msgs/Aidetection
"""
YOLO_DETECTION_TOPIC = '/yolo_tflite_data'


"""
DETECTION_ACTION : str

The action to take when a valid detection is received.

Available actions:

    'hold'
        Switch the PX4 vehicle to Hold mode.

    'land'
        Command the vehicle to land.

Future actions can be added to the action registry below.
"""
DETECTION_ACTION = 'hold'

"""
TFLITE_OUTPUT_PIPE_PREFIX: str

The prefix of the output pipe created by voxl-tflite-server.
"""
TFLITE_OUTPUT_PIPE_PREFIX = 'yolo'

# =========================================================


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
        detection_topic=YOLO_DETECTION_TOPIC,
        action_callback=None,
        pipe_prefix=TFLITE_OUTPUT_PIPE_PREFIX
    )

    px4_connector = PX4Connector(
        logger=yolo_receiver.get_logger(),
        system_address=PX4_ADDRESS
    )

    actions = {

        'hold': px4_connector.set_hold_mode,

        'land': px4_connector.set_land_mode,

    }

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