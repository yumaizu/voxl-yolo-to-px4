import asyncio
import os
import subprocess
from time import sleep

from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

from voxl_msgs.msg import Aidetection

class TFLiteReceiver(Node):

    def __init__(
        self,
        loop,
        detection_topic,
        action_callback,
        pipe_prefix,
        logger,
        log_all_detections=False
    ):

        super(TFLiteReceiver, self).__init__(
            'tflite_receiver'
        )

        self.loop = loop
        self.action_callback = action_callback
        self.action_triggered = False
        self.logger = logger
        self.log_all_detections = log_all_detections

        # -------------------------------------------------
        # Start voxl-tflite-server service
        # -------------------------------------------------

        self.logger.info(
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

        # wait for voxl-tflite-server to start outputting a pipe
        pipe_path = "/run/mpa/{}tflite".format(pipe_prefix + "_" if pipe_prefix else "")
        self.logger.info(
            'Waiting for voxl-tflite-server to start...'
        )
        
        # Wait for directory to be created AND contain files (like 'info', 'request')
        while True:
            if os.path.exists(pipe_path) and os.path.isdir(pipe_path):
                if len(os.listdir(pipe_path)) > 0:
                    break
            sleep(1)

        self.logger.info(
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

        self.logger.info(
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
        # Subscribe to TFLite detections
        # -------------------------------------------------

        self.subscription = self.create_subscription(
            Aidetection,
            detection_topic,
            self.detection_callback,
            qos_profile
        )

        self.logger.info(
            'Listening for TFLite detections on {}'.format(
                detection_topic
            )
        )

    def detection_callback(self, msg):

        if self.log_all_detections:
            self.logger.info(
                'Detection: {} | Confidence: {:.2f}'.format(
                    msg.class_name,
                    msg.class_confidence
                )
            )

        if (
            msg.class_name == 'person'
            and msg.class_confidence >= 0.6
            and not self.action_triggered
        ):

            self.logger.warning(
                'PERSON DETECTED - executing configured action'
            )

            self.action_triggered = True

            asyncio.run_coroutine_threadsafe(
                self.action_callback(),
                self.loop
            )

    def destroy_node(self):

        self.logger.info(
            'Stopping voxl_mpa_to_ros2'
        )

        if self.mpa_to_ros_process:

            self.mpa_to_ros_process.terminate()

            try:

                self.mpa_to_ros_process.wait(
                    timeout=5
                )

            except subprocess.TimeoutExpired:

                self.logger.warning(
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

        self.logger.info(
            'Stopping voxl-tflite-server service'
        )
        
        # Wait for voxl-tflite-server to stop before destroying the node
        stat = subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
        while stat == 0:    # should return 3 when inactive
            self.logger.info(
                'Waiting for voxl-tflite-server to stop...'
            )
            stat = subprocess.run(['systemctl', 'is-active', '--quiet', 'voxl-tflite-server']).returncode
            sleep(1)

        super().destroy_node()
