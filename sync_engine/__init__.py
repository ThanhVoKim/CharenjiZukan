# -*- coding: utf-8 -*-
"""
sync_engine package - TTS-Video Sync (Chunk-Based Stretch) Core Logic
"""

from sync_engine.models import SubBlock, TimelineSegment
from sync_engine.analyzer import (
    filter_tts_subtitles,
    classify_and_compute_slots,
    compute_speeds,
    build_timeline_map,
    remap_timestamp,
)
from sync_engine.video_processor import (
    query_keyframes,
    snap_to_nearest_keyframe,
    process_video_chunks_parallel,
)
from sync_engine.audio_assembler import assemble_audio_track
from sync_engine.timestamp_remapper import recalculate_srt, recalculate_ass
from sync_engine.renderer import render_final_video

__all__ = [
    "SubBlock",
    "TimelineSegment",
    "filter_tts_subtitles",
    "classify_and_compute_slots",
    "compute_speeds",
    "build_timeline_map",
    "remap_timestamp",
    "query_keyframes",
    "snap_to_nearest_keyframe",
    "process_video_chunks_parallel",
    "assemble_audio_track",
    "recalculate_srt",
    "recalculate_ass",
    "render_final_video",
]
