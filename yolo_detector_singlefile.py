#!/usr/bin/env python3

import asyncio
import threading

import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

from voxl_msgs.msg import Aidetection

from mavsdk import System
from mavsdk.action import ActionError


class YOLODetector(Node):

    def __init__(self, loop):
        super(YOLODetector, self).__init__('yolo_detector')

        self.loop = loop

        self.hold_triggered = False
        self.drone = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            Aidetection,
            '/yolo_tflite_data',
            self.detection_callback,
            qos_profile
        )

        self.get_logger().info(
            'Listening for YOLO detections on /yolo_tflite_data'
        )

    def detection_callback(self, msg):

        self.get_logger().info(
            'Detection: {} | Confidence: {}'.format(
                msg.class_name,
                msg.class_confidence
            )
        )

        if (
            msg.class_name == 'person'
            and msg.class_confidence >= 0.6
            and not self.hold_triggered
        ):

            self.get_logger().warn(
                'PERSON DETECTED - switching to HOLD'
            )

            self.hold_triggered = True

            # Run the async MAVSDK function on the asyncio event loop
            asyncio.run_coroutine_threadsafe(
                self.set_hold_mode(),
                self.loop
            )

    async def set_hold_mode(self):

        try:

            self.get_logger().warn(
                'Sending Hold mode command to PX4'
            )

            await self.drone.action.hold()

            self.get_logger().warn(
                'PX4 Hold command successfully sent'
            )

        except ActionError as e:

            self.get_logger().error(
                'Failed to send Hold command: {}'.format(e)
            )

        except Exception as e:

            self.get_logger().error(
                'Unexpected MAVSDK error: {}'.format(e)
            )


async def connect_to_px4(node):

    node.get_logger().info(
        'Connecting to PX4 using MAVSDK on udp://:14551'
    )

    drone = System()

    await drone.connect(
        system_address='udp://:14551'
    )

    node.get_logger().info(
        'Waiting for PX4 connection...'
    )

    async for state in drone.core.connection_state():

        if state.is_connected:

            node.get_logger().info(
                'Connected to PX4!'
            )

            break

    node.drone = drone

    return drone


def main():

    # ---------------------------------------------------------
    # Start the asyncio event loop in a separate thread
    # ---------------------------------------------------------

    loop = asyncio.new_event_loop()

    asyncio_thread = threading.Thread(
        target=loop.run_forever,
        daemon=True
    )

    asyncio_thread.start()

    # ---------------------------------------------------------
    # Initialize ROS 2
    # ---------------------------------------------------------

    rclpy.init()

    node = YOLODetector(loop)

    # ---------------------------------------------------------
    # Connect to PX4 using MAVSDK
    # ---------------------------------------------------------

    future = asyncio.run_coroutine_threadsafe(
        connect_to_px4(node),
        loop
    )

    try:

        # Wait for MAVSDK connection to complete
        future.result()

        node.get_logger().info(
            'YOLO detector is ready'
        )

        # Start ROS 2 event loop
        rclpy.spin(node)

    except KeyboardInterrupt:

        node.get_logger().info(
            'Shutting down'
        )

    finally:

        node.destroy_node()

        rclpy.shutdown()

        loop.call_soon_threadsafe(
            loop.stop
        )

        asyncio_thread.join()


if __name__ == '__main__':

    main()