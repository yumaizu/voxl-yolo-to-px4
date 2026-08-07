import asyncio
import cv2
import time
import datetime
import torch
import numpy as np
from ultralytics import YOLO
from DelayedVideo import DelayedVideo

class YOLOProcessor:
    def __init__(self, loop, rtsp_url, model_path, confidence_threshold, device, 
                 enable_debug_window, web_view, artificial_latency_ms, action_callback, logger):
        self.loop = loop
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.enable_debug_window = enable_debug_window
        self.web_view = web_view
        
        # Convert milliseconds to seconds for MAVLink asyncio.sleep
        self.latency_sec = float(artificial_latency_ms) / 1000.0
        
        self.action_callback = action_callback
        self.logger = logger
        self.action_triggered = False
        self.running = True

        # Initialize the delayed video stream buffer
        self.video = DelayedVideo(rtsp_url, artificial_latency_ms, logger)

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

    def _format_ts(self, timestamp):
        """Converts epoch float to human-readable YYYY-MM-DD HH:MM:SS.fff"""
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def start(self):
        # Start the background RTSP reader and buffering thread
        self.video.start()
        
        # --- FULLSCREEN WINDOW INITIALIZATION ---
        if self.enable_debug_window:
            cv2.namedWindow("YOLO Debug Window", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("YOLO Debug Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
        # Variables for FPS calculation
        fps_ema = 0.0
        prev_loop_time = time.time()
            
        while self.running:
            # pre_yolo_time is captured the exact moment the buffer yields the delayed frame
            ret, frame, frame_received_time = self.video.read()
            pre_yolo_time = time.time() 
            
            if not ret:
                # Buffer is still filling or waiting for delay to expire; yield CPU
                time.sleep(0.01)
                
                # Keep OpenCV GUI responsive even when waiting
                if self.enable_debug_window:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Exit requested via 'q' key.")
                        break
                continue

            # YOLO inference
            results = self.model(frame, device=self.device, verbose=False)
            inference_complete_time = time.time()

            person_detected = False
            for result in results:
                for box in result.boxes:
                    if int(box.cls) == 0 and float(box.conf) >= self.confidence_threshold:
                        person_detected = True
                        break
                if person_detected:
                    break

            if person_detected and not self.action_triggered:
                self.action_triggered = True
                
                self.logger.warning(f"[{self._format_ts(frame_received_time)}] PERSON DETECTED: RTSP frame received (Pre-YOLO Delay START)")
                self.logger.warning(f"[{self._format_ts(pre_yolo_time)}] PERSON DETECTED: Sent to YOLO (Pre-YOLO Delay END)")
                self.logger.warning(f"[{self._format_ts(inference_complete_time)}] PERSON DETECTED: YOLO inference complete (Pre-MAVLink Delay START)")
                
                # Virtual Latency Injection: After YOLO, before MAVLink
                async def delayed_mavlink_action():
                    if self.latency_sec > 0:
                        await asyncio.sleep(self.latency_sec)
                    
                    mavlink_send_time = time.time()
                    self.logger.warning(f"[{self._format_ts(mavlink_send_time)}] PERSON DETECTED: Sending MAVLink signal (Pre-MAVLink Delay END)")
                    await self.action_callback()

                asyncio.run_coroutine_threadsafe(
                    delayed_mavlink_action(),
                    self.loop
                )

            # Generate annotated debug frame
            annotated_frame = results[0].plot()

            # --- FPS Calculation and Drawing ---
            current_time = time.time()
            time_diff = current_time - prev_loop_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                if fps_ema == 0.0:
                    fps_ema = current_fps
                else:
                    fps_ema = (fps_ema * 0.9) + (current_fps * 0.1) # Smooth out the FPS display
            prev_loop_time = current_time

            # Draw the FPS counter in the top-left corner
            cv2.putText(annotated_frame, f"FPS: {fps_ema:.1f}", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            # Update JPEG frame buffer for web streaming if enabled
            if self.web_view:
                ret_jpeg, jpeg = cv2.imencode('.jpg', annotated_frame)
                if ret_jpeg:
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
        self.video.stop()
        if self.enable_debug_window:
            cv2.destroyAllWindows()