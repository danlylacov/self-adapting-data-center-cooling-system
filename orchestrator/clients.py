"""HTTP clients for DC Digital Twin and ML services (sync httpx)."""

from __future__ import annotations

from typing import Any, Optional

import httpx


class HttpError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TwinClient:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, json: Any = None) -> Any:
        try:
            r = self._client.request(method, path, json=json)
        except httpx.RequestError as e:
            raise HttpError(f"{method} {path}: {e}") from e
        if r.status_code >= 400:
            detail = r.text
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            raise HttpError(f"{method} {path}: {detail}", status_code=r.status_code)
        if r.status_code == 204 or not r.content:
            return None
        try:
            return r.json()
        except ValueError as e:
            raise HttpError(f"{method} {path}: invalid JSON: {e}") from e

    def post(self, path: str, body: Optional[dict] = None) -> Any:
        return self._request("POST", path, json=body or {})

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def health(self) -> Any:
        return self.get("/health")


class MlClient:
    """Generic GET/POST for ML microservices."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        try:
            r = self._client.get(path, params=params)
        except httpx.RequestError as e:
            raise HttpError(f"GET {path}: {e}") from e
        if r.status_code >= 400:
            raise HttpError(f"GET {path}: {r.text}", status_code=r.status_code)
        return r.json()

    def post(self, path: str, body: dict) -> Any:
        try:
            r = self._client.post(path, json=body)
        except httpx.RequestError as e:
            raise HttpError(f"POST {path}: {e}") from e
        if r.status_code >= 400:
            raise HttpError(f"POST {path}: {r.text}", status_code=r.status_code)
        try:
            return r.json()
        except ValueError as e:
            raise HttpError(f"POST {path}: invalid JSON: {e}") from e

    def health(self) -> Any:
        return self.get("/health")
