"""Dependency-light local web frontend for forecast-conditioned separation."""
from __future__ import annotations

import argparse
import base64
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from .inference import SeparationEngine, wav_bytes


WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}


def make_handler(engine: SeparationEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OceanSeparator/1.0"

        def _headers(self, status: int, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self' blob:; style-src 'self'; script-src 'self'")
            self.end_headers()

        def _json(self, status: int, value: dict) -> None:
            payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._headers(status, "application/json; charset=utf-8")
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/api/status":
                self._json(200, {
                    "ready": True,
                    "variant": engine.config["variant"],
                    "sample_rate": engine.sample_rate,
                    "history_steps": engine.history_steps,
                    "requires_history": engine.requires_history,
                    "validation_scope": engine.config.get(
                        "current_background_definition", engine.config.get("background_waveform", "surrogate background")
                    ),
                    "device": str(engine.device),
                    "checkpoint": engine.checkpoint_path.name,
                })
                return
            routes = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
            filename = routes.get(self.path)
            if filename is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = WEB_ROOT / filename
            self._headers(200, CONTENT_TYPES[path.suffix])
            self.wfile.write(path.read_bytes())

        def do_POST(self):
            if self.path != "/api/separate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 180 * 1024 * 1024:
                    raise ValueError("request must be between 1 byte and 180 MiB")
                request = json.loads(self.rfile.read(length))
                audio = engine.read_audio(base64.b64decode(request["audio_base64"], validate=True))
                history_text = request.get("history_csv", "")
                history_audio = request.get("history_audio_base64", "")
                if history_audio:
                    raw_history = engine.read_audio(base64.b64decode(history_audio, validate=True))
                    history = engine.history_from_audio(raw_history)
                else:
                    history = engine.parse_history_csv(history_text) if history_text else None
                if engine.requires_history and history is None:
                    raise ValueError("this forecast-conditioned checkpoint requires clean historical audio or event-masked TOL CSV")
                result = engine.separate(audio, history)
                self._json(200, {
                    "source_wav": base64.b64encode(wav_bytes(result.pop("source"), engine.sample_rate)).decode("ascii"),
                    "ocean_wav": base64.b64encode(wav_bytes(result.pop("ocean"), engine.sample_rate)).decode("ascii"),
                    "sample_rate": engine.sample_rate,
                    **result,
                })
            except Exception as error:
                self._json(400, {"error": str(error)})

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/practical_film.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device")
    args = parser.parse_args()
    engine = SeparationEngine(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"Ocean separator ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
