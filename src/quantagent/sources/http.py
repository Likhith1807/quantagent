"""Small stdlib HTTP helper with retries (no extra dependency for the fetch scripts)."""
import json
import time
import urllib.error
import urllib.request

USER_AGENT = "quantagent-fetch/1.0 (research use; one request at a time per worker)"


def get(url: str, timeout: float = 60, retries: int = 4, backoff: float = 2.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs
                return r.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def get_json(url: str, **kw):
    return json.loads(get(url, **kw).decode("utf-8"))
