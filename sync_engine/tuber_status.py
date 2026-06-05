"""
sync_engine/tuber_status.py
===========================
Quản lý status.json từng group (Phase R/S). Status đơn giản:
pending/running/done/failed/skipped; chi tiết lỗi ở failedStep + lastError.
V3 thêm inputHash cho resume skip-done.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# failedStep dự kiến (plan)
STEP_BUILDING_SPEECH_AUDIO = "building_speech_audio"
STEP_EXPORTING_MANIFEST = "exporting_manifest"
STEP_RENDERING_OVERLAY = "rendering_overlay"
STEP_COMPOSITING = "compositing"
STEP_VALIDATING = "validating"
STEP_CONCATENATING = "concatenating"
STEP_CLEANUP = "cleanup"


def new_status(group_id: str) -> Dict[str, Any]:
    return {
        "groupId": group_id,
        "status": STATUS_PENDING,
        "currentStep": None,
        "failedStep": None,
        "attempts": 0,
        "lastError": None,
        "fallbackTriggered": False,
        "inputHash": None,
    }


def write_status(group_dir: Path, status: Dict[str, Any]) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / "status.json"
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_status(group_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(group_dir) / "status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_group_input_hash(
    group_manifest: dict,
    prerender_manifest: Optional[dict],
    stretched_video: Path,
) -> str:
    """SHA-256 hash các yếu tố quyết định output 1 group.

    Gồm: segments, renderStartFrame, renderDurationFrames, compOffsetX/Y,
    prerender assetId + outputWidth/Height, và (mtime_ns, size) của stretched_video.

    Returns:
        Hex digest string.
    """
    payload = {
        "segments": [
            {
                "startFrame": s.get("startFrame"),
                "endFrame": s.get("endFrame"),
                "blockType": s.get("blockType"),
                "hasTts": s.get("hasTts"),
                "mouthEvents": s.get("mouthEvents"),
            }
            for s in (group_manifest.get("segments") or [])
        ],
        "renderStartFrame": group_manifest.get("renderStartFrame"),
        "renderDurationFrames": group_manifest.get("renderDurationFrames"),
        "compOffsetX": group_manifest.get("compOffsetX"),
        "compOffsetY": group_manifest.get("compOffsetY"),
        "compWidth": group_manifest.get("compWidth"),
        "compHeight": group_manifest.get("compHeight"),
        "fps": group_manifest.get("fps"),
        "prerender": None if not prerender_manifest else {
            "assetId": prerender_manifest.get("assetId"),
            "outputWidth": prerender_manifest.get("outputWidth"),
            "outputHeight": prerender_manifest.get("outputHeight"),
        },
    }
    h = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    try:
        st = Path(stretched_video).stat()
        h.update(f"vid:{st.st_mtime_ns}:{st.st_size}".encode())
    except OSError:
        pass
    return h.hexdigest()
