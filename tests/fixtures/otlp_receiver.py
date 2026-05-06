"""In-process OTLP receiver — captures emitted spans for assertion.

Wire level: bodies are kept as raw bytes so `test_sentinel_fuzz` can scan
them. Decode level: bodies are parsed back into dicts using the same proto
library the backend uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


@dataclass
class Capture:
    bodies: list[bytes] = field(default_factory=list)
    spans: list[dict[str, Any]] = field(default_factory=list)

    def all_attrs(self) -> list[dict[str, Any]]:
        """Return a flat list of {attr_key: value} dicts, one per captured span."""
        out: list[dict[str, Any]] = []
        for batch in self.spans:
            for rs in batch.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for span in ss.get("spans", []):
                        flat: dict[str, Any] = {"_name": span.get("name")}
                        for a in span.get("attributes", []):
                            flat[a["key"]] = _unwrap_value(a["value"])
                        out.append(flat)
        return out


def _unwrap_value(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "arrayValue" in v:
        return [_unwrap_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def make_capture_app(capture: Capture) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/traces")
    async def traces(request: Request) -> dict[str, str]:
        body = await request.body()
        capture.bodies.append(bytes(body))
        req = ExportTraceServiceRequest()
        req.ParseFromString(body)
        capture.spans.append(
            MessageToDict(req, preserving_proto_field_name=False)
        )
        return {"status": "ok"}

    return app


def capture_http_post(capture: Capture) -> tuple:
    """Return a (`http_post`, `cleanup`) pair.

    Uses `fastapi.testclient.TestClient`, which runs the ASGI app on a
    thread-driven sync bridge — compatible with our sync emitter.
    """
    from fastapi.testclient import TestClient

    app = make_capture_app(capture)
    client = TestClient(app)

    def post(url: str, *, content=None, headers=None, timeout=None, **_):
        path = url
        if "://" in url:
            path = "/" + url.split("/", 3)[3]
        return client.post(path, content=content, headers=headers)

    def close() -> None:
        client.close()

    return post, close
