import asyncio
import os
import subprocess
import datetime
import cv2
from time import sleep

from rclpy.node import Node

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import HistoryPolicy

from voxl_msgs.msg import Aidetection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class TFLiteReceiver(Node):

    def __init__(
        self,
        loop,
        action_callback,
        pipe_prefix,
        confidence_threshold,
        logger,
        log_all_detections=False,
        image_save_dir='/data/yolo'
    ):

        super(TFLiteReceiver, self).__init__(
            'tflite_receiver'
        )

        self.loop = loop
        self.action_callback = action_callback
        self.action_triggered = False
        self.confidence_threshold = float(confidence_threshold)
        self.logger = logger
        self.log_all_detections = log_all_detections
        self.image_save_dir = image_save_dir
        
        # Dynamically generate the topic strings from the pipe prefix
        self.detection_topic = f"/{pipe_prefix}_tflite_data" if pipe_prefix else "/tflite_data"
        self.image_topic = f"/{pipe_prefix}_tflite" if pipe_prefix else "/tflite"

        # Image buffering for saving exact detection frames
        self.bridge = CvBridge()
        self.latest_frame = None

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
        # Subscribe to TFLite images (for saving exact frames)
        # -------------------------------------------------

        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile
        )

        self.logger.info(
            'Listening for TFLite images on {}'.format(
                self.image_topic
            )
        )

        # -------------------------------------------------
        # Subscribe to TFLite detections
        # -------------------------------------------------

        self.subscription = self.create_subscription(
            Aidetection,
            self.detection_topic,
            self.detection_callback,
            qos_profile
        )

        self.logger.info(
            'Listening for TFLite detections on {}'.format(
                self.detection_topic
            )
        )

    def image_callback(self, msg):
        """Continually buffer the latest visual frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass # Suppress image conversion errors to prevent console spam

    def reset_trigger(self):
        """Resets the detection block to allow subsequent triggers."""
        self.action_triggered = False
        self.logger.info('Action trigger state has been reset.')

    def detection_callback(self, msg):

        if self.log_all_detections:
            self.logger.debug(
                'Detection: {} | Confidence: {:.2f}'.format(
                    msg.class_name,
                    msg.class_confidence
                )
            )

        if (
            msg.class_name == 'person'
            and msg.class_confidence >= self.confidence_threshold
            and not self.action_triggered
        ):

            self.logger.warning(
                'PERSON DETECTED - executing configured action'
            )

            self.action_triggered = True
            
            # Save the exact frame buffer
            if self.latest_frame is not None:
                ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = os.path.join(self.image_save_dir, f"detection_{ts_str}.jpg")
                cv2.imwrite(filename, self.latest_frame)
                self.logger.info(f"Saved exact detection frame to {filename}")
            else:
                self.logger.warning("Person detected, but image frame not yet available to save.")

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