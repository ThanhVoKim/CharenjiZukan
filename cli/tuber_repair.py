#!/usr/bin/env python3
"""
cli/tuber_repair.py
===================
Late repair / re-render job cho tuber overlay (Phase repair).

Tình huống: `sync_video` đã chạy với tuber enabled nhưng tuber fail → fallback
render final NON-tuber. artifactPolicy=repairable đã giữ lại base video, mixed
audio, final_render_inputs, run_manifest + group_manifest. CLI này render tuber
overlay muộn rồi gọi lại render_final_video từ base đã có tuber.

Cách gọi:
    python -m cli.tuber_repair --tuber-root tuber-output/<job>/tuber

KHÔNG nhồi thêm flag vào sync_video — đây là job riêng (theo plan).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from utils.logger import get_logger, setup_logging
from sync_engine.renderer import render_final_video
from sync_engine.tuber_config import parse_tuber_config_dict
from sync_engine.tuber_manifest import load_run_manifest
from sync_engine.tuber_artifacts import (
    load_final_render_manifest, deserialize_image_overlay_events,
)
from sync_engine.tuber_overlay import (
    render_groups_to_video, jobs_from_run_manifest, TuberOverlayError,
)

logger = get_logger("tuber_repair")


def run_repair(tuber_root: Path, output_path: str = None) -> str:
    tuber_root = Path(tuber_root)
    run_manifest = load_run_manifest(tuber_root)

    cfg = parse_tuber_config_dict(run_manifest.get("tuberConfig", {}))
    ap = run_manifest.get("artifactPolicy") or cfg.artifact_policy()

    media_dir = Path(run_manifest["mediaDir"])
    final_render_inputs_dir = tuber_root / "final_render_inputs"
    logs_dir = tuber_root / "logs"
    tmp_dir = tuber_root / "repair_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Validate input bắt buộc
    base_video = Path(run_manifest["baseVideo"])
    final_audio = Path(run_manifest["finalAudio"])
    fr_manifest = load_final_render_manifest(final_render_inputs_dir)
    for p in (base_video, final_audio):
        if not p.exists():
            raise FileNotFoundError(f"Repair thiếu artifact bắt buộc: {p}")

    # Resolve prerender manifest (V3)
    prerender_manifest = None
    prerender_dir = None
    pm_path = run_manifest.get("prerenderManifest")
    if pm_path and Path(pm_path).exists():
        try:
            prerender_manifest = json.loads(Path(pm_path).read_text(encoding="utf-8"))
            prerender_dir = Path(pm_path).parent
        except (json.JSONDecodeError, OSError):
            prerender_manifest = None
    use_prerender = prerender_manifest is not None

    # Resolve overlay_format: đọc từ run_manifest (ghi lúc all-in), fallback config (V4)
    overlay_format = run_manifest.get("overlayFormat") or cfg.overlay_format

    # Stable anchor cho inputHash (resume): video gốc. Fallback base_video nếu
    # run_manifest cũ chưa có sourceVideo (hash sẽ khác → re-render, an toàn).
    src_video = run_manifest.get("sourceVideo")
    source_video = Path(src_video) if src_video else base_video

    # Render tuber overlay muộn (group base.mp4 đã sẵn từ all-in run)
    remotion = run_manifest["remotion"]
    asset = run_manifest["asset"]
    jobs = jobs_from_run_manifest(run_manifest)
    if not jobs:
        raise TuberOverlayError("run_manifest không có group nào để repair.")

    video_with_tuber = Path(run_manifest["videoWithTuber"])
    render_groups_to_video(
        project_dir=Path(remotion["projectDir"]),
        asset_id=asset["assetId"],
        asset_dir=Path(asset["assetDir"]),
        chromakey=cfg.chromakey,
        groups=jobs,
        output_video=video_with_tuber,
        tmp_dir=tmp_dir,
        logs_dir=logs_dir,
        retry_attempts=cfg.retry_attempts,
        artifact_policy=ap,
        use_prerender=use_prerender,
        prerender_dir=prerender_dir,
        prerender_manifest=prerender_manifest,
        do_prepare_assets=not use_prerender,
        stretched_video=base_video,
        source_video=source_video,
        max_workers=cfg.max_workers,
        skip_done=cfg.resume_skip_done,
        overlay_format=overlay_format,
    )

    # Final render từ base đã có tuber + final_render_inputs
    suffix = cfg.repair_output_suffix
    out_name = fr_manifest.get("output_name", "video_synced")
    if output_path is None:
        output_path = str(media_dir.parent / f"{out_name}{suffix}.mp4")

    image_events = None
    img_events_path = fr_manifest.get("image_overlay_events")
    if img_events_path and Path(img_events_path).exists():
        data = json.loads(Path(img_events_path).read_text(encoding="utf-8"))
        image_events = deserialize_image_overlay_events(data)

    render_config = {}
    rc_path = fr_manifest.get("render_config")
    if rc_path and Path(rc_path).exists():
        render_config = json.loads(Path(rc_path).read_text(encoding="utf-8"))

    render_final_video(
        stretched_video=str(video_with_tuber),
        mixed_audio=str(final_audio),
        subtitle_synced_srt=fr_manifest.get("subtitle_synced_srt"),
        output_path=output_path,
        note_overlay_synced_ass=fr_manifest.get("note_overlay_synced_ass"),
        render_config=render_config,
        use_gpu=fr_manifest.get("use_gpu", True),
        image_overlay_events=image_events,
    )
    logger.info("Repair hoàn tất → %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Tuber overlay late repair / re-render")
    parser.add_argument("--tuber-root", required=True,
                        help="Đường dẫn tuberRoot (vd tuber-output/<job>/tuber)")
    parser.add_argument("--output", default=None,
                        help="Override path final output (mặc định: <job>/<name><suffix>.mp4)")
    args = parser.parse_args()

    setup_logging()
    tuber_root = Path(args.tuber_root)
    if not tuber_root.is_absolute():
        tuber_root = PROJECT_ROOT / tuber_root
    try:
        run_repair(tuber_root, args.output)
    except Exception as e:
        logger.exception("Repair lỗi: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
