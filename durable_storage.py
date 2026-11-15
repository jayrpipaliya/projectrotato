"""Durable cloud storage for Double Edge (Vercel Blob / local fallback)."""
import os
import json
import logging
import urllib.request
import urllib.parse
from pathlib import Path

logger = logging.getLogger("double_edge.storage")
BLOB_ENDPOINT = "https://blob.vercel-storage.com"


def get_blob_token():
    return os.environ.get("BLOB_READ_WRITE_TOKEN")


def blob_list(prefix=""):
    token = get_blob_token()
    if not token:
        return []
    url = BLOB_ENDPOINT
    if prefix:
        url += f"?prefix={urllib.parse.quote(prefix)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("blobs", [])
    except Exception as e:
        logger.warning(f"[Storage] Failed to list blobs: {e}")
        return []


def blob_put(pathname, content, content_type="application/octet-stream"):
    token = get_blob_token()
    if not token:
        return None
    url = f"{BLOB_ENDPOINT}/{pathname}"
    if isinstance(content, str):
        content = content.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=content,
        headers={
            "Authorization": f"Bearer {token}",
            "x-vercel-blob-access": "private",
            "x-add-random-suffix": "0",
            "x-allow-overwrite": "1",
            "Content-Type": content_type,
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"[Storage] Failed to put blob {pathname}: {e}")
        return None


def blob_get(download_url):
    token = get_blob_token()
    if not token:
        return None
    req = urllib.request.Request(download_url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()
    except Exception as e:
        logger.warning(f"[Storage] Failed to get blob {download_url}: {e}")
        return None


def sync_cloud_to_local(runtime_dir: Path, raw_dir: Path = None):
    """Restore persisted files from Vercel Blob on startup/cold start."""
    token = get_blob_token()
    if not token:
        return 0
    blobs = blob_list()
    restored = 0
    for b in blobs:
        pathname = b.get("pathname", "")
        url = b.get("url", "")
        if not pathname or not url:
            continue
        try:
            target = None
            if pathname == "gamma_history.json" or pathname in ("spy_rsp.csv", "qqq_rsp.csv", "iwm_qqq.csv"):
                target = runtime_dir / pathname
            elif raw_dir and pathname.endswith("_us_d.csv"):
                target = raw_dir / pathname

            if target:
                content = blob_get(url)
                if content:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    restored += 1
        except Exception as e:
            logger.warning(f"[Storage] Error restoring {pathname}: {e}")
    return restored


def persist_file_to_cloud(local_path: Path, cloud_pathname: str = None):
    """Upload a local file to durable cloud storage."""
    token = get_blob_token()
    if not token or not local_path.is_file():
        return False
    cloud_name = cloud_pathname or local_path.name
    content_type = "application/json" if cloud_name.endswith(".json") else "text/csv"
    try:
        data = local_path.read_bytes()
        res = blob_put(cloud_name, data, content_type)
        return bool(res)
    except Exception as e:
        logger.warning(f"[Storage] Error persisting {local_path}: {e}")
        return False


def persist_gamma_history(runtime_dir: Path):
    """Upload latest gamma_history.json to durable cloud storage."""
    gamma_file = runtime_dir / "gamma_history.json"
    if gamma_file.is_file():
        return persist_file_to_cloud(gamma_file, "gamma_history.json")
    return False
