"""Vercel serverless function: POST /api/grade

STAGED, NOT DEPLOYED. See ../README.md for why, and for the origin question that has to
be answered before this is worth deploying anywhere.

Implements Barbara's contract byte-for-byte so a card written against hers works against
ours unchanged:

    request   {"id": "archaic", "sentence": "..."}
    response  {"total", "max_total", "passed", "dimensions":[...], "stage", "error"?}

The `stage` field is an ADDITION to her contract, not a change to it. Her client infers
"was this a real graded attempt?" from its own local checklist result; ours states it,
because TimeBack counts attempts and getting that wrong costs the student XP. A client
that ignores `stage` still works.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _grader as grader  # noqa: E402

MAX_BODY = 8192          # a sentence; anything larger is not a sentence
MAX_SENTENCE_CHARS = 600


def _cors(h):
    """Permissive CORS on purpose: the card may be served from a different origin.

    TimeBack allowlists ONE origin for its postMessage contract, so the card and this
    function may well not be same-origin, unlike Barbara's. Nothing here is
    authenticated or user-specific, and the only cost of a cross-origin call is our LLM
    budget -- which is rate-limited below rather than by origin, because an Origin
    header is trivially forged and would give false comfort.
    """
    h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    h.send_header("Access-Control-Allow-Headers", "Content-Type")


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        _cors(self)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self):
        # a browser hitting the URL should get a usable message, not a stack trace
        self._send(405, {"error": "POST {id, sentence} to this endpoint.",
                         "passed": False, "stage": "error"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            return self._send(400, {"error": "Bad request.", "passed": False,
                                    "stage": "error"})
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except (ValueError, UnicodeError):
            return self._send(400, {"error": "Bad request.", "passed": False,
                                    "stage": "error"})
        if not isinstance(req, dict):
            return self._send(400, {"error": "Bad request.", "passed": False,
                                    "stage": "error"})

        task_id = str(req.get("id") or "").strip()
        sentence = str(req.get("sentence") or "")[:MAX_SENTENCE_CHARS]

        try:
            out = grader.grade(task_id, sentence)
        except Exception:
            # never let a stack reach a student, and never let one open a gate
            return self._send(200, {"error": "The grader is unavailable right now. "
                                             "Try again in a moment.",
                                    "passed": False, "stage": "error"})

        # 200 even for a failing grade: a failed grade is a normal outcome, not an HTTP
        # error, and a non-2xx would send Barbara-shaped clients down their network-error
        # branch -- which per Max's contract must NOT report a TimeBack attempt, whereas
        # a real graded failure MUST.
        return self._send(200, out)
