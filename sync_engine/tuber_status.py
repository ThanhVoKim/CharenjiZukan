"""
sync_engine/tuber_status.py
===========================
Quản lý status.json từng group (Phase R/S). Status đơn giản:
pending/running/done/failed/skipped; chi tiết lỗi ở failedStep + lastError.
"""
from __future__ import annotations

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
