"""Vercel serverless function: GET /api/task?id=archaic

STAGED, NOT DEPLOYED.

Mirrors Barbara's open GET /api/task, whose payload shape we adopted rather than invented:
{id, word, task, check_label, check_keywords, hints}. Her `hints` field is what carries
Sarah's "here's another word you might want to use" idea, and Lucia's 2026-08-27 decision
is that it -- with the definition -- appears only AFTER a first failed try.

WHAT THIS DELIBERATELY DOES NOT RETURN
--------------------------------------
`definition` and `avoid_verbatim`. The definition is what the grader marks against; the
task's whole design is that the student writes the sentence WITHOUT it in front of them,
so serving it from an open endpoint would hand the answer to anyone who opened the
network tab. `avoid_verbatim` is the anti-paste list, and publishing it publishes the
exact strings to avoid. Both stay server-side. `hint_available` tells the card whether a
hint EXISTS without revealing it.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _grader as grader  # noqa: E402

PUBLIC = ("word", "task", "check_label", "check_keywords")


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        tid = (q.get("id") or [""])[0].strip()
        try:
            tasks = grader._load("tasks.json")
        except Exception:
            return self._send(500, {"error": "task store unavailable"})
        t = tasks.get(tid)
        if t is None:
            return self._send(404, {"error": "unknown task id"})
        out = {"id": tid}
        out.update({k: t.get(k) for k in PUBLIC})
        out["hint_available"] = bool(t.get("hints"))
        return self._send(200, out)


