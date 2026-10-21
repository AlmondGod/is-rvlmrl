"""Local scripted HTTP fixture for transport/physics regression, NOT a model."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time

SEQUENCE = [(kind, i) for i in range(4) for kind in ("position_member", "release_member")]
SEQUENCE += [("park_grippers", None)]
SEQUENCE += [("nail_corner", i) for i in range(4)]
SEQUENCE += [("feed_panel", None), ("position_panel", None), ("release_panel", None)]
SEQUENCE += [("nail_panel", i) for i in range(4)]
SEQUENCE += [("flip_structure", None), ("finish", None)]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        observation = json.loads(request["messages"][-1]["content"][0]["text"])["observation"]
        n = len(observation["command_history"])
        sequence = SEQUENCE[:8] + [("stop", None)] if self.server.frame_only else SEQUENCE
        skill, index = sequence[min(n, len(sequence) - 1)] if self.server.full else ("stop", None)
        time.sleep(self.server.delay)
        text = json.dumps({"skill": skill, "index": index,
                           "reason": "SCRIPTED HARNESS TEST; not an image inference."})
        body = json.dumps({"model": "HARNESS_TEST_ONLY", "id": "mock",
                          "choices": [{"finish_reason": "stop", "message": {"content": text}}],
                          "usage": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--full", action="store_true")
    p.add_argument("--frame-only", action="store_true")
    p.add_argument("--delay", type=float, default=0.)
    args = p.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.full = args.full
    server.frame_only = args.frame_only
    server.delay = args.delay
    print("SCRIPTED TEST SERVER", server.server_address, flush=True)
    server.serve_forever()
