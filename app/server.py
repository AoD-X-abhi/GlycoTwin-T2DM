import os
import sys
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        directory = os.path.dirname(os.path.abspath(__file__))
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers for smooth mobile preview
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def run_server(port=8080):
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, CustomHandler)
    except OSError:
        port = 8081
        server_address = ('', port)
        httpd = HTTPServer(server_address, CustomHandler)

    local_ip = get_local_ip()
    print("=" * 65)
    print("  GlycoTwin Mobile Digital Twin Server Running!")
    print(f"  Local Desktop URL:  http://localhost:{port}")
    print(f"  Mobile Phone Access: http://{local_ip}:{port}")
    print("  (Ensure your mobile phone is on the same Wi-Fi network)")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port)
