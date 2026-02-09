#!/usr/bin/env python3
"""
CytoAtlas Development Server

Production-ready local server for the CytoAtlas visualization platform.
Features:
- CORS headers for local development
- Static file serving with caching
- Health check endpoint
- Configurable port
- Graceful shutdown
"""

import argparse
import http.server
import json
import mimetypes
import os
import signal
import socketserver
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

# Configuration
DEFAULT_PORT = 8080
DEFAULT_HOST = "localhost"
CACHE_MAX_AGE = 3600  # 1 hour for static assets
NO_CACHE_EXTENSIONS = {'.html', '.json', '.js'}  # Don't cache these during development

# MIME type additions
mimetypes.add_type('application/json', '.json')
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')


class CytoAtlasRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS, caching, and health check."""

    lite_mode = False  # Set via class attribute before server starts

    def __init__(self, *args, directory=None, **kwargs):
        self.server_start_time = datetime.now().isoformat()
        super().__init__(*args, directory=directory, **kwargs)

    def handle(self):
        """Override to suppress broken-pipe / connection-reset errors."""
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        """Custom log format with timestamps."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = format % args
        print(f"[{timestamp}] {self.address_string()} - {message}")

    def send_cors_headers(self):
        """Send CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Max-Age', '86400')

    def send_cache_headers(self, path):
        """Send appropriate caching headers based on file type."""
        ext = Path(path).suffix.lower()
        if ext in NO_CACHE_EXTENSIONS:
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
        else:
            self.send_header('Cache-Control', f'public, max-age={CACHE_MAX_AGE}')

    def send_security_headers(self):
        """Send security-related headers."""
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')

    def end_headers(self):
        """Override to add custom headers."""
        self.send_cors_headers()
        self.send_security_headers()
        super().end_headers()

    def do_OPTIONS(self):
        """Handle preflight CORS requests."""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests with special endpoints."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        # Lite mode: rewrite /data/ requests to /data_lite/
        if self.lite_mode and path.startswith('/data/'):
            self.path = self.path.replace('/data/', '/data_lite/', 1)

        # Health check endpoint
        if path == '/api/health':
            self.send_health_response()
            return

        # API endpoint for listing available data files
        if path == '/api/data':
            self.send_data_listing()
            return

        # Serve index.html for root
        if path == '/':
            self.path = '/index.html'

        # Default file serving
        super().do_GET()

    def send_health_response(self):
        """Send health check response."""
        health_data = {
            'status': 'healthy',
            'server': 'CytoAtlas Dev Server',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'uptime_since': self.server_start_time,
            'lite_mode': self.lite_mode,
        }
        self.send_json_response(health_data)

    def send_data_listing(self):
        """List available data files in the data directory."""
        subdir = 'data_lite' if self.lite_mode else 'data'
        data_dir = Path(self.directory) / subdir
        if data_dir.exists():
            files = [f.name for f in data_dir.iterdir() if f.is_file()]
            self.send_json_response({'files': sorted(files), 'source': subdir})
        else:
            self.send_json_response({'files': [], 'error': f'{subdir} directory not found'})

    def send_json_response(self, data, status=200):
        """Send a JSON response."""
        content = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_cache_headers('.json')
        self.end_headers()
        self.wfile.write(content)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded HTTP server for handling concurrent requests."""
    allow_reuse_address = True
    daemon_threads = True

    def handle_error(self, request, client_address):
        """Suppress noisy tracebacks from client disconnects."""
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def run_server(host, port, directory, lite=False):
    """Run the development server."""
    # Ensure directory exists
    directory = Path(directory).resolve()
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if lite:
        lite_dir = directory / 'data_lite'
        if not lite_dir.exists():
            print(f"Error: Lite data directory '{lite_dir}' does not exist.")
            print("  Run: python scripts/create_data_lite.py")
            sys.exit(1)
        CytoAtlasRequestHandler.lite_mode = True

    # Change to the serving directory
    os.chdir(directory)

    # Create handler with directory
    handler = partial(CytoAtlasRequestHandler, directory=str(directory))

    # Create server
    server = ThreadedHTTPServer((host, port), handler)

    # Setup graceful shutdown
    def signal_handler(signum, frame):
        print("\n\nShutting down server...")
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print startup message
    mode_label = "LITE MODE" if lite else "FULL DATA"
    print("=" * 60)
    print(f"  CytoAtlas Development Server ({mode_label})")
    print("=" * 60)
    print(f"  Serving from: {directory}")
    if lite:
        print(f"  Data source:  data_lite/ (rewriting /data/ -> /data_lite/)")
    print(f"  Local URL:    http://{host}:{port}")
    print(f"  Health check: http://{host}:{port}/api/health")
    print(f"  Data listing: http://{host}:{port}/api/data")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Run server
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CytoAtlas Development Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python dev-server.py                    # Serve current directory on port 8080
  python dev-server.py --lite             # Serve with lite data (~100 MB vs 8.7 GB)
  python dev-server.py -p 3000            # Use port 3000
  python dev-server.py -d ./visualization # Serve specific directory
  python dev-server.py --host 0.0.0.0     # Allow external connections
        '''
    )
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=DEFAULT_PORT,
        help=f'Port to serve on (default: {DEFAULT_PORT})'
    )
    parser.add_argument(
        '--host',
        default=DEFAULT_HOST,
        help=f'Host to bind to (default: {DEFAULT_HOST})'
    )
    parser.add_argument(
        '-d', '--directory',
        default='.',
        help='Directory to serve (default: current directory)'
    )
    parser.add_argument(
        '--lite',
        action='store_true',
        help='Serve lite data (rewrites /data/ requests to /data_lite/)'
    )

    args = parser.parse_args()

    run_server(args.host, args.port, args.directory, lite=args.lite)


if __name__ == '__main__':
    main()
