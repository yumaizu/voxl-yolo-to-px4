import socket
import time
import datetime

class VisionHubConnector:
    def __init__(self, logger, drone_ip, port=5005):
        """
        :param logger: Your existing logger instance
        :param drone_ip: The local network IP address of the Starling 2 (e.g., '192.168.8.1')
        :param port: The UDP port the C script is listening on
        """
        self.logger = logger
        self.drone_ip = drone_ip
        self.port = port
        self.sock = None

    def _format_ts(self, timestamp):
        """Converts epoch float to human-readable YYYY-MM-DD HH:MM:SS.fff"""
        return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def connect(self):
        """Initializes the UDP socket. No handshake required for connectionless UDP."""
        self.logger.info(f'Initializing UDP connection to VOXL Vision Hub at {self.drone_ip}:{self.port}')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logger.info('Vision Hub connector ready!')

    def trigger_descent(self):
        """Fires the LAND trigger over the network."""
        try:
            formatted_time = self._format_ts(time.time())
            self.logger.warning(f'[{formatted_time}] PERSON DETECTED: Sending Offboard descent trigger to Vision Hub')
            
            # Send a tiny 4-byte packet. If it drops, the loop will just send another one next frame.
            self.sock.sendto(b"LAND", (self.drone_ip, self.port))
            
        except Exception as e:
            self.logger.error(f'Failed to send descent trigger: {e}')