"""Local-only web interface. Created 2026-09-21 KST."""
import argparse
import base64
import binascii
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, UnidentifiedImageError
from recognition import Recognizer

class Handler(BaseHTTPRequestHandler):
    recognizer = None

    def respond(self, status, body, content_type='application/json; charset=utf-8'):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/':
            self.respond(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/health':
            self.respond(200, {'ready': True})
        else:
            self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        if self.path != '/predict':
            self.respond(404, {'error': 'Not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1_000_000:
                raise ValueError('Invalid request size.')
            if self.headers.get_content_type() != 'application/json':
                raise ValueError('JSON required.')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict) or not isinstance(body.get('image'), str):
                raise ValueError('Image required.')
            prefix, separator, encoded = body['image'].partition(',')
            if prefix != 'data:image/png;base64' or not separator:
                raise ValueError('PNG image required.')
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.width > 1024 or image.height > 1024:
                    raise ValueError('Image must be PNG and at most 1024 x 1024.')
                rgba = image.convert('RGBA')
                white = Image.new('RGBA', rgba.size, 'white')
                white.alpha_composite(rgba)
                result = self.recognizer.predict(white.convert('L'))
            self.respond(200, result)
        except (ValueError, KeyError, TypeError, binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            self.respond(400, {'error': str(exc)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    Handler.recognizer = Recognizer()
    server = HTTPServer(('127.0.0.1', args.port), Handler)
    server.timeout = 1
    print(f'Open http://127.0.0.1:{args.port}  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
