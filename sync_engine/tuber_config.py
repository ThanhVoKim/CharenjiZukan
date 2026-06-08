"""
sync_engine/tuber_config.py
===========================
Load + validate tuber overlay config và resolve output layout.

Dùng chung cho `cli/sync_video.py` (all-in path) và `cli/tuber_repair.py`
(late-repair path). Module này KHÔNG đụng tới FFmpeg/Remotion — chỉ là config +
path resolution thuần để dễ unit-test (Layer 1).

Quyết định plan liên quan:
  - Phase A: load + validate key tối thiểu.
  - Phase D: resolve jobName (M3 sentinel "video_synced"), tuberRoot.
  - artifactPolicy.mode = repairable (default) + override chi tiết (Phase T).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sync_video")

# Sentinel default của --output-name trong sync_video (M3): coi như "chưa đặt tên".
DEFAULT_OUTPUT_NAME_SENTINEL = "video_synced"
DEFAULT_TUBER_OUTPUT_DIR = "tuber-output"

# artifactPolicy mặc định khi mode=repairable (Phase T).
_REPAIRABLE_DEFAULTS = {
    "overlayFrames": "safe",      # safe = xoá sau khi composite+validate OK
    "finalRenderInputs": "keep",
    "logs": "keep",
    "failedGroups": "keep",
}

# Các key tối thiểu bắt buộc khi enabled=true (Phase A bước 5).
_REQUIRED_KEYS = [
    "asset.assetDir",
    "asset.mouthTrack",
    "asset.mouthSprites",
    "asset.bodySource",
    "grouping.maxGroupSec",
    "overlay.format",
    "retry.retryAttempts",
    "retry.onExhausted",
]

# Các key Remotion — chỉ bắt buộc khi overlay.mode != "prerender".
_REMOTION_REQUIRED_KEYS = [
    "remotion.projectDir",
    "remotion.compositionId",
    "remotion.entryPoint",
]


class TuberConfigError(ValueError):
    """Config tuber không hợp lệ (thiếu key bắt buộc, giá trị sai)."""


def _get_nested(d: Dict[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


@dataclass
class TuberConfig:
    """Config tuber đã parse + đã resolve path layout."""

    enabled: bool
    raw: Dict[str, Any] = field(default_factory=dict)

    # Path layout (resolve khi gọi resolve_layout)
    project_root: Optional[Path] = None
    job_name: Optional[str] = None
    tuber_base_dir: Optional[Path] = None
    tuber_root: Optional[Path] = None
    media_dir: Optional[Path] = None
    groups_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    final_render_inputs_dir: Optional[Path] = None

    # ── Config accessors (đọc từ raw với default) ──
    @property
    def output_dir(self) -> str:
        return self.raw.get("outputDir") or DEFAULT_TUBER_OUTPUT_DIR

    @property
    def max_group_sec(self) -> float:
        return float(_get_nested(self.raw, "grouping.maxGroupSec") or 300.0)

    @property
    def overlay_format(self) -> str:
        fmt = _get_nested(self.raw, "overlay.format") or "direct"
        if fmt not in ("direct", "png_sequence"):
            logger.warning("overlay.format=%r không hợp lệ → dùng 'direct'.", fmt)
            return "direct"
        return fmt

    @property
    def overlay_mode(self) -> str:
        """Mode kiểm soát render: 'remotion' | 'prerender' | 'auto' (default).

        'remotion' → luôn dùng Remotion, bỏ qua prerender_manifest.json.
        'prerender' → bắt buộc dùng pre-render, lỗi nếu không có manifest.
        'auto' → auto-detect: dùng prerender nếu có manifest, ngược lại Remotion.
        """
        mode = _get_nested(self.raw, "overlay.mode")
        if mode in ("remotion", "prerender", "auto"):
            return mode
        return "auto"

    @property
    def mouth_mode(self) -> str:
        return _get_nested(self.raw, "mouth.mode") or "cue"

    @property
    def mouth_silence_db(self) -> float:
        return float(_get_nested(self.raw, "mouth.silenceDb") or -40.0)

    @property
    def mouth_min_silence_ms(self) -> float:
        return float(_get_nested(self.raw, "mouth.minSilenceMs") or 200.0)

    @property
    def mouth_cadence_ms(self) -> float:
        return float(_get_nested(self.raw, "mouth.cadenceMs") or 150.0)

    @property
    def mouth_peak_margin(self) -> float:
        """Tầng 2 — ngưỡng phát hiện đỉnh sóng (env chuẩn hoá [0,1])."""
        v = _get_nested(self.raw, "mouth.peakMargin")
        return float(v) if v is not None else 0.02

    @property
    def mouth_min_vowel_interval_ms(self) -> float:
        """Tầng 2 — cooldown (ms) giữa 2 lần đổi khẩu hình nguyên âm."""
        v = _get_nested(self.raw, "mouth.minVowelIntervalMs")
        return float(v) if v is not None else 120.0

    @property
    def mouth_vowel_low_pct(self) -> float:
        """Tầng 2 — percentile centroid → U_TH (thấp = 'u'/う)."""
        v = _get_nested(self.raw, "mouth.vowelLowPercentile")
        return float(v) if v is not None else 20.0

    @property
    def mouth_vowel_high_pct(self) -> float:
        """Tầng 2 — percentile centroid → E_TH (cao = 'e'/え)."""
        v = _get_nested(self.raw, "mouth.vowelHighPercentile")
        return float(v) if v is not None else 80.0

    @property
    def mouth_states(self) -> list:
        states = _get_nested(self.raw, "mouth.mouthStates")
        return list(states) if states else ["closed", "half", "open"]

    @property
    def max_workers(self) -> int:
        return int(_get_nested(self.raw, "performance.maxWorkers") or 2)

    @property
    def resume_skip_done(self) -> bool:
        v = _get_nested(self.raw, "resume.skipDone")
        return True if v is None else bool(v)

    @property
    def debug_frame_output_enabled(self) -> bool:
        return bool(_get_nested(self.raw, "debug.frameOutput.enabled") or False)

    @property
    def debug_frame_margin(self) -> int:
        return int(_get_nested(self.raw, "debug.frameOutput.marginFrames") or 3)

    @property
    def prerender_character_dir(self) -> Optional[Path]:
        """Path to prerendered character dir (có thể relative). None = chưa pre-render."""
        cd = _get_nested(self.raw, "asset.prerender.characterDir")
        if cd:
            return self._abs(str(cd))
        # Default: assetDir / prerendered
        return self.asset_dir() / "prerendered"

    @property
    def retry_attempts(self) -> int:
        return int(_get_nested(self.raw, "retry.retryAttempts") or 0)

    @property
    def on_exhausted(self) -> str:
        return _get_nested(self.raw, "retry.onExhausted") or "render_without_tuber"

    @property
    def repair_output_suffix(self) -> str:
        return _get_nested(self.raw, "repair.defaultOutputSuffix") or "_with_tuber"

    @property
    def character(self) -> Dict[str, Any]:
        return dict(self.raw.get("character") or {})

    @property
    def chromakey(self) -> Dict[str, Any]:
        return dict(_get_nested(self.raw, "asset.chromakey") or {})

    def artifact_policy(self) -> Dict[str, str]:
        """Resolve artifactPolicy thành dict đầy đủ các sub-key.

        mode=repairable → áp default _REPAIRABLE_DEFAULTS, cho phép override.
        mode khác → giữ nguyên override, default về "keep" cho an toàn.
        """
        ap = dict(self.raw.get("artifactPolicy") or {})
        mode = ap.get("mode", "repairable")
        if mode == "repairable":
            resolved = dict(_REPAIRABLE_DEFAULTS)
        else:
            resolved = {k: "keep" for k in _REPAIRABLE_DEFAULTS}
        for k in _REPAIRABLE_DEFAULTS:
            if k in ap:
                resolved[k] = ap[k]
        resolved["mode"] = mode
        return resolved

    # ── Asset paths (absolute, từ project_root nếu relative) ──
    def _abs(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute() and self.project_root is not None:
            path = self.project_root / path
        return path

    def asset_dir(self) -> Path:
        return self._abs(_get_nested(self.raw, "asset.assetDir"))

    def mouth_track_path(self) -> Path:
        rel = _get_nested(self.raw, "asset.mouthTrack") or "mouth_track.json"
        p = Path(rel)
        return p if p.is_absolute() else self.asset_dir() / p

    def mouth_dir(self) -> Path:
        """Thư mục chứa mouth sprites (lấy từ dir của sprite đầu tiên trong mouthSprites)."""
        sprites = _get_nested(self.raw, "asset.mouthSprites") or {}
        for rel in sprites.values():
            p = Path(rel)
            d = (self.asset_dir() / p).parent if not p.is_absolute() else p.parent
            return d
        return self.asset_dir() / "mouth"

    def body_transparent_dir(self) -> Path:
        """Thư mục body-transparent (PNG frames sau chromakey). Convention: assetDir/body-transparent."""
        return self.asset_dir() / "body-transparent"

    def asset_id(self) -> str:
        """assetId = tên thư mục asset (dùng cho public/pngtuber/<id>)."""
        explicit = _get_nested(self.raw, "asset.assetId")
        if explicit:
            return str(explicit)
        return self.asset_dir().name

    def remotion_project_dir(self) -> Path:
        return self._abs(_get_nested(self.raw, "remotion.projectDir"))

    def resolve_layout(self, project_root: Path, *, input_video: Optional[str],
                       output_name: Optional[str]) -> "TuberConfig":
        """Phase D: tính jobName + tuberRoot và set các path. Trả về chính self."""
        self.project_root = Path(project_root)

        # jobName (M3): output_name != sentinel default → dùng nó; else stem video.
        if output_name and output_name != DEFAULT_OUTPUT_NAME_SENTINEL:
            self.job_name = output_name
        elif input_video:
            self.job_name = Path(input_video).stem
        else:
            self.job_name = DEFAULT_OUTPUT_NAME_SENTINEL

        out_dir = Path(self.output_dir)
        self.tuber_base_dir = out_dir if out_dir.is_absolute() else self.project_root / out_dir
        self.tuber_root = self.tuber_base_dir / self.job_name / "tuber"
        self.media_dir = self.tuber_root / "media"
        self.groups_dir = self.tuber_root / "groups"
        self.logs_dir = self.tuber_root / "logs"
        self.final_render_inputs_dir = self.tuber_root / "final_render_inputs"
        return self

    def make_dirs(self) -> None:
        for d in (self.media_dir, self.groups_dir, self.logs_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)


def _validate_required(raw: Dict[str, Any]) -> List[str]:
    """Trả về list key thiếu (Phase A bước 5)."""
    missing = []
    for key in _REQUIRED_KEYS:
        if _get_nested(raw, key) in (None, ""):
            missing.append(key)
    # Remotion keys chỉ bắt buộc khi overlay.mode != "prerender"
    overlay_mode = _get_nested(raw, "overlay.mode")
    if overlay_mode != "prerender":
        for key in _REMOTION_REQUIRED_KEYS:
            if _get_nested(raw, key) in (None, ""):
                missing.append(key)
    return missing


def load_tuber_config(config_path: Optional[str], project_root: Path) -> TuberConfig:
    """Phase A: load config từ path.

    - config_path None/rỗng → disabled.
    - enabled=false → disabled.
    - enabled=true → validate; thiếu key bắt buộc → raise TuberConfigError.
    """
    if not config_path:
        return TuberConfig(enabled=False)

    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = project_root / cfg_file
    if not cfg_file.exists():
        raise TuberConfigError(f"Không tìm thấy tuber config: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not raw.get("enabled", False):
        logger.info("Tuber config enabled=false → bỏ qua tuber overlay.")
        return TuberConfig(enabled=False, raw=raw)

    missing = _validate_required(raw)
    if missing:
        raise TuberConfigError(
            "Tuber config thiếu key bắt buộc khi enabled=true: " + ", ".join(missing)
        )
    return TuberConfig(enabled=True, raw=raw)


def parse_tuber_config_dict(raw: Dict[str, Any]) -> TuberConfig:
    """Dùng cho repair: dựng TuberConfig từ dict đã đọc (run_manifest mang config)."""
    enabled = bool(raw.get("enabled", True))
    if enabled:
        missing = _validate_required(raw)
        if missing:
            raise TuberConfigError(
                "Tuber config (repair) thiếu key: " + ", ".join(missing)
            )
    return TuberConfig(enabled=enabled, raw=raw)
