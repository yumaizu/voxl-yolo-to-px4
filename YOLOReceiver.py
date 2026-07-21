import asyncio
import os
import subprocess
from time import sleep

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
        action_callback,
        pipe_prefix
    ):

        super(YOLOReceiver, self).__init__(
            'yolo_receiver'
        )

        self.loop = loop

        self.action_callback = action_callback

        self.action_triggered = False

        # -------------------------------------------------
        # Start voxl-tflite-server service
        # -------------------------------------------------

        self.get_logger().info(
            'Starting voxl-tflite-server service'
        )

        subprocess.Popen(
            [
                'systemctl',
                'start',
                'voxl-tflite-server'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # # Wait for voxl-tflite-server to start before starting voxl_mpa_to_ros2
        # stat = subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
        # while stat != 0:
        #     self.get_logger().info(
        #         'Waiting for voxl-tflite-server to start...'
        #     )
        #     stat =subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
        #     sleep(1)

        # wait for voxl-tflite-server to start outputting a pipe
        pipe_path = "/run/mpa/{}tflite".format(pipe_prefix + "_" if pipe_prefix else "")
        self.get_logger().info(
            'Waiting for voxl-tflite-server to start...'
        )
        # wait for directory to be created
        while not os.path.exists(pipe_path):
            sleep(1)

        self.get_logger().info(
            'Started voxl-tflite-server service'
        )

        # -------------------------------------------------
        # Start voxl_mpa_to_ros2
        # -------------------------------------------------

        self.mpa_to_ros_process = subprocess.Popen(
            [
                'ros2',
                'run',
                'voxl_mpa_to_ros2',
                'voxl_mpa_to_ros2_node'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self.get_logger().info(
            'Started voxl_mpa_to_ros2'
        )

        # -------------------------------------------------
        # ROS 2 QoS configuration
        # -------------------------------------------------

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # -------------------------------------------------
        # Subscribe to YOLO detections
        # -------------------------------------------------

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

            asyncio.run_coroutine_threadsafe(
                self.action_callback(),
                self.loop
            )

    def destroy_node(self):

        self.get_logger().info(
            'Stopping voxl_mpa_to_ros2'
        )

        if self.mpa_to_ros_process:

            self.mpa_to_ros_process.terminate()

            try:

                self.mpa_to_ros_process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:

                self.get_logger().warn(
                    'voxl_mpa_to_ros2 did not terminate gracefully'
                )

                self.mpa_to_ros_process.kill()

        subprocess.Popen(
            [
                'systemctl',
                'stop',
                'voxl-tflite-server'
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self.get_logger().info(
            'Stopping voxl-tflite-server service'
        )
        # Wait for voxl-tflite-server to stop before destroying the node
        stat =subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
        while stat == 0:    # should return 3 when inactive
            self.get_logger().info(
                'Waiting for voxl-tflite-server to stop...'
            )
            stat =subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
            sleep(1)

        super().destroy_node()