import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True

class StreamHandler(BaseHTTPRequestHandler):
    """Serves MJPEG video stream over HTTP."""
    server_instance = None

    def do_GET(self):
        if self.path in ['/', '/stream']:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while self.server_instance and self.server_instance.running:
                    jpeg_bytes = self.server_instance.get_latest_jpeg()
                    if jpeg_bytes is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(jpeg_bytes)))
                        self.end_headers()
                        self.wfile.write(jpeg_bytes)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # Cap web stream at ~30 FPS
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP access logging to keep console clean
        return


class YOLOWebView:
    def __init__(self, port, logger):
        self.port = int(port)
        self.logger = logger
        self.running = False
        self.latest_jpeg = None
        self.frame_lock = threading.Lock()
        self.httpd = None
        self.web_thread = None

    def update_frame(self, jpeg_bytes):
        """Called by YOLOProcessor to supply the newest frame."""
        with self.frame_lock:
            self.latest_jpeg = jpeg_bytes

    def get_latest_jpeg(self):
        with self.frame_lock:
            return self.latest_jpeg

    def _start_server(self):
        try:
            StreamHandler.server_instance = self
            self.httpd = ThreadedHTTPServer(('0.0.0.0', self.port), StreamHandler)
            self.logger.info(f"Web stream available at http://0.0.0.0:{self.port}")
            self.httpd.serve_forever()
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")

    def start(self):
        self.running = True
        self.web_thread = threading.Thread(target=self._start_server, daemon=True)
        self.web_thread.start()

    def stop(self):
        self.running = False
        if self.httpd:
            self.httpd.shutdown()
        if self.web_thread and self.web_thread.is_alive():
            self.web_thread.join(timeout=1.0)