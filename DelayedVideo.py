import os
import cv2
import time
import threading
from collections import deque

class DelayedVideo:
    def __init__(self, rtsp_url, delay_ms, logger):
        self.rtsp_url = rtsp_url
        self.delay_sec = float(delay_ms) / 1000.0
        self.logger = logger
        
        self.running = False
        self.cap = None
        self.buffer = deque()
        self.reader_thread = None

    def _connect(self):
        """Attempts connection to RTSP stream."""
        while self.running:
            self.logger.info(f"Attempting connection to RTSP stream at: {self.rtsp_url}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Prevent FFMPEG internal network buffer buildup
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.logger.info("Successfully connected to RTSP feed!")
                return True
                
            self.logger.warning("RTSP server rejected connection. Retrying in 1 second...")
            time.sleep(1)
        return False

    def _reader_loop(self):
        """Background thread that continuously consumes frames to prevent decoder corruption."""
        if not self._connect():
            return

        while self.running and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if not self.running:
                        break
                    self.logger.error("Lost frame from RTSP stream. Waiting 2 seconds before reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    if not self._connect():
                        break
                    continue
                
                # Append frame and the exact time it was received
                self.buffer.append((frame, time.time()))
                
            except cv2.error:
                if not self.running:
                    break
                else:
                    self.logger.error("OpenCV C++ exception in reader loop.")
                    break
            except Exception as e:
                if not self.running:
                    break
                self.logger.error(f"Unexpected error in reader loop: {e}")
                break

    def start(self):
        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def read(self):
        """
        Retrieves the frame that matches the artificial delay setting.
        Automatically drops stale frames if a backlog accumulates.
        Returns: (success_boolean, frame, frame_timestamp)
        """
        if not self.buffer:
            return False, None, 0.0

        now = time.time()

        # Maximum allowed frame age before it is considered backlog (Target delay + 100ms tolerance)
        max_allowed_age = self.delay_sec + 0.100

        # 1. PURGE BACKLOG: Drop any frames that are older than our allowed threshold
        while len(self.buffer) > 1 and (now - self.buffer[0][1]) > max_allowed_age:
            self.buffer.popleft()

        # 2. EVALUATE TARGET FRAME
        oldest_frame, oldest_time = self.buffer[0]
        frame_age = now - oldest_time

        # If the frame has reached our target delay, yield it
        if frame_age >= self.delay_sec:
            self.buffer.popleft()
            return True, oldest_frame, oldest_time

        # Still waiting for frame to reach target delay
        return False, None, 0.0

    def stop(self):
        self.running = False
        
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()