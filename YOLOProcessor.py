import asyncio
import cv2
import time
import datetime
import torch
import os
import threading
import numpy as np
from ultralytics import YOLO
from DelayedVideo import DelayedVideo

# Suppress harmless OpenCV/Qt font warnings
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'

# Suppress FFmpeg HEVC/H.265 stream decoding error spam in the console
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;32"  # AV_LOG_INFO (or 24 for warning)

class YOLOProcessor:
    def __init__(self, loop, rtsp_url, model_path, confidence_threshold, device, 
                 enable_debug_window, web_view, artificial_latency_ms, action_callback, shutdown_callback, logger, log_all_detections=False, image_save_dir='./yolo_detections'):
        self.loop = loop
        self.model_path = model_path
        self.confidence_threshold = float(confidence_threshold)
        self.enable_debug_window = enable_debug_window
        self.web_view = web_view
        self.log_all_detections = log_all_detections
        self.image_save_dir = image_save_dir
        self.shutdown_callback = shutdown_callback

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

    def _get_four_digit_ms_str(self, epoch_time=None):
        """Generates timestamp string with 4-digit millisecond precision (e.g. _5685) to match logs."""
        if epoch_time:
            dt = datetime.datetime.fromtimestamp(epoch_time)
        else:
            dt = datetime.datetime.now()
        base_str = dt.strftime('%Y%m%d_%H%M%S_%f')
        # %f gives 6 digits (microseconds). Taking the first 5 characters of the fractional part gives 4 digits of milliseconds.
        # Format: YYYYmmdd_HHMMSS + _ + 4 digits
        return f"{dt.strftime('%Y%m%d_%H%M%S')}_{dt.strftime('%f')[:4]}"

    def reset_trigger(self):
        """Resets the detection block to allow subsequent triggers."""
        self.action_triggered = False
        self.logger.info('Action trigger state has been reset.')

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
                time.sleep(0.005)
                if self.enable_debug_window:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Exit requested via 'q' key.")
                        self.running = False
                        break
                continue

            # YOLO inference
            results = self.model(frame, device=self.device, verbose=False)
            inference_complete_time = time.time()

            # Generate annotated debug frame early so it can be saved on detection
            annotated_frame = results[0].plot()

            person_detected = False
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    
                    if self.log_all_detections:
                        cls_name = self.model.names[cls_id] if hasattr(self.model, 'names') else str(cls_id)
                        self.logger.debug(f"Detection: {cls_name} | Confidence: {conf:.2f}")

                    if cls_id == 0 and conf >= self.confidence_threshold:
                        person_detected = True
                        if not self.log_all_detections:
                            break # Early exit optimization if we aren't logging every box
                
                if person_detected and not self.log_all_detections:
                    break

            if person_detected and not self.action_triggered:
                self.action_triggered = True
                
                self.logger.warning(f"[{self._format_ts(frame_received_time)}] PERSON DETECTED: RTSP frame received (Pre-YOLO Delay START)")
                self.logger.warning(f"[{self._format_ts(pre_yolo_time)}] PERSON DETECTED: Sent to YOLO (Pre-YOLO Delay END)")
                self.logger.warning(f"[{self._format_ts(inference_complete_time)}] PERSON DETECTED: YOLO inference complete (Pre-MAVLink Delay START)")
                
                # Setup structured directory path using the 4-digit ms format matching the received frame log
                rtsp_ts_folder = self._get_four_digit_ms_str(frame_received_time)
                target_dir = os.path.join(self.image_save_dir, rtsp_ts_folder)
                os.makedirs(target_dir, exist_ok=True)
                
                # Save the initial trigger frame immediately
                init_filename = os.path.join(target_dir, f"frame_{rtsp_ts_folder}.jpg")
                cv2.imwrite(init_filename, annotated_frame)

                # Thread coordination event to signal when action finishes + buffer
                action_finished_event = threading.Event()

                def background_saver():
                    try:
                        saved_count = 1
                        # Continuously harvest frames as fast as possible without missing any
                        while not action_finished_event.is_set():
                            r_ret, r_frame, r_time = self.video.read()
                            if r_ret:
                                ts_sub = self._get_four_digit_ms_str(r_time if r_time else None)
                                f_name = os.path.join(target_dir, f"frame_{ts_sub}.jpg")
                                cv2.imwrite(f_name, r_frame)
                                saved_count += 1
                            else:
                                time.sleep(0.002) # brief yield if buffer is temporarily empty
                            
                        # +1000ms grace period buffer after MAVLink execution completes
                        buffer_end_time = time.time() + 1.0
                        while time.time() < buffer_end_time:
                            r_ret, r_frame, r_time = self.video.read()
                            if r_ret:
                                ts_sub = self._get_four_digit_ms_str(r_time if r_time else None)
                                f_name = os.path.join(target_dir, f"frame_{ts_sub}.jpg")
                                cv2.imwrite(f_name, r_frame)
                                saved_count += 1
                            else:
                                time.sleep(0.002)

                        self.logger.info(f"Finished saving frame sequence. Total frames saved: {saved_count} in {target_dir}")
                    except Exception as e:
                        self.logger.error(f"Background saver thread error: {e}")

                # Launch background thread
                saver_thread = threading.Thread(target=background_saver, daemon=True)
                saver_thread.start()

                async def delayed_mavlink_action():
                    if self.latency_sec > 0:
                        await asyncio.sleep(self.latency_sec)
                    
                    mavlink_send_time = time.time()
                    self.logger.warning(f"[{self._format_ts(mavlink_send_time)}] PERSON DETECTED: Sending MAVLink signal (Pre-MAVLink Delay END)")
                    await self.action_callback()
                    
                    # Signal background thread that MAVLink execution is complete
                    action_finished_event.set()

                asyncio.run_coroutine_threadsafe(
                    delayed_mavlink_action(),
                    self.loop
                )

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
                    self.running = False
                    break

        # Trigger full application shutdown cleanly from the main loop
        if self.shutdown_callback:
            self.loop.call_soon_threadsafe(self.shutdown_callback)

    def stop(self):
        self.logger.info("Stopping YOLO processor...")
        self.running = False
        self.video.stop()
        if self.enable_debug_window:
            cv2.destroyAllWindows()