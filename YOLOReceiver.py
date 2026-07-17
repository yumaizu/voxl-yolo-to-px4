import asyncio

from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

from voxl_msgs.msg import Aidetection


class YOLOReceiver(Node):

    def __init__(
        self,
        loop,
        detection_topic,
        action_callback
    ):

        super(YOLOReceiver, self).__init__(
            'yolo_receiver'
        )

        self.loop = loop

        self.action_callback = action_callback

        self.action_triggered = False

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(
            Aidetection,
            detection_topic,
            self.detection_callback,
            qos_profile
        )

        self.get_logger().info(
            'Listening for YOLO detections on {}'.format(
                detection_topic
            )
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
            and not self.action_triggered
        ):

            self.get_logger().warn(
                'PERSON DETECTED - executing configured action'
            )

            self.action_triggered = True

            # Run the asynchronous action callback
            # on the MAVSDK asyncio event loop.

            asyncio.run_coroutine_threadsafe(
                self.action_callback(),
                self.loop
            )