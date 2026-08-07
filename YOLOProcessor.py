import os
import asyncio
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO

class YOLOProcessor:
    def __init__(self, loop, rtsp_url, model_path, confidence_threshold, device, 
                 enable_debug_window, web_view, action_callback, logger):
        self.loop = loop
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.enable_debug_window = enable_debug_window
        self.web_view = web_view
        self.action_callback = action_callback
        self.logger = logger

        self.action_triggered = False
        self.running = True
        self.cap = None

        # Dynamic device handling
        if device.lower() == 'auto':
            self.device = '0' if torch.cuda.is_available() else 'cpu'
            self.logger.info(f"Auto-detected device: {self.device}")
        else:
            self.device = device
            self.logger.info(f"Using configured device: {self.device}")
        
        self.logger.info(f"Loading YOLO model on device: {self.device}...")
        self.model = YOLO(self.model_path)
        
        self.logger.info("Running model warmup...")
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.model(blank_frame, device=self.device, verbose=False)

    def _connect_with_retries(self):
        """Attempts connection to RTSP stream."""
        while self.running:
            self.logger.info(f"Attempting connection to RTSP stream at: {self.rtsp_url}")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                self.logger.info("Successfully connected to RTSP feed!")
                return True
                
            self.logger.warning("RTSP server rejected connection. Retrying in 1 second...")
            time.sleep(1)
            
        return False

    def start(self):
        if not self._connect_with_retries():
            return
            
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Lost frame from RTSP stream. Waiting 2 seconds before reconnecting...")
                self.cap.release()
                time.sleep(2)
                if not self._connect_with_retries():
                    break
                continue

            frame_received_time = time.time()

            # YOLO inference
            results = self.model(frame, device=self.device, verbose=False)
            inference_complete_time = time.time()

            person_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0 and box.conf >= self.confidence_threshold:
                        person_detected = True
                        break
                if person_detected:
                    break

            if person_detected and not self.action_triggered:
                self.action_triggered = True
                
                self.logger.warning(f"[{frame_received_time}] PERSON DETECTED: RTSP frame received")
                self.logger.warning(f"[{inference_complete_time}] PERSON DETECTED: YOLO inference complete")
                
                asyncio.run_coroutine_threadsafe(
                    self.action_callback(),
                    self.loop
                )

            # Generate annotated debug frame
            annotated_frame = results[0].plot()

            # Update JPEG frame buffer for web streaming if enabled
            if self.web_view:
                ret, jpeg = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    self.web_view.update_frame(jpeg.tobytes())

            # Render local desktop window if enabled
            if self.enable_debug_window:
                cv2.imshow("YOLO Debug Window", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Exit requested via 'q' key.")
                    break

    def stop(self):
        self.logger.info("Stopping YOLO processor...")
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.enable_debug_window:
            cv2.destroyAllWindows()