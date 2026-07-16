"""Local mock webhook endpoint. A real game backend would expose an HTTP
endpoint that receives churn-intervention triggers; this just logs whatever
it receives so we can prove the scorer -> webhook call actually happened.

Usage: .venv10projects/bin/python webhook/receiver.py [port]
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body.decode(errors="replace")}

        print(f"[webhook-receiver] POST {self.path} -> {json.dumps(payload)}", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "received"}')

    def log_message(self, fmt, *args):
        pass  # ponytail: suppress default access-log noise, we print our own line above


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), WebhookHandler)
    print(f"[webhook-receiver] listening on http://127.0.0.1:{PORT}/webhook", flush=True)
    server.serve_forever()
