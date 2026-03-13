# Utils module for CharenjiZukan

from utils.srt_parser import parse_srt, parse_srt_file, ts_to_ms, segments_to_srt
from utils.audio_utils import load_audio, export_audio, create_silence
from utils.ass_utils import (
    srt_timestamp_to_ass,
    ass_timestamp_to_srt,
    wrap_text,
    normalize_newlines,
    create_dialogue_line,
    parse_ass_file,
    write_ass_file,
    convert_srt_segments_to_ass_dialogues,
)
from utils.media_utils import (
    detect_media_type,
    scale_time_ms,
    check_rubberband_available,
    stretch_audio_rubberband,
    stretch_audio_atempo,
    stretch_audio,
    change_video_speed,
    scale_srt_timestamps,
    scale_ass_timestamps,
    parse_ass_timestamp_to_ms,
    ms_to_ass_timestamp,
    get_default_output_path,
)

__all__ = [
    # SRT Parser
    'parse_srt',
    'parse_srt_file',
    'ts_to_ms',
    'segments_to_srt',
    # Audio Utils
    'load_audio',
    'export_audio',
    'create_silence',
    # ASS Utils
    'srt_timestamp_to_ass',
    'ass_timestamp_to_srt',
    'wrap_text',
    'normalize_newlines',
    'create_dialogue_line',
    'parse_ass_file',
    'write_ass_file',
    'convert_srt_segments_to_ass_dialogues',
    # Media Utils
    'detect_media_type',
    'scale_time_ms',
    'check_rubberband_available',
    'stretch_audio_rubberband',
    'stretch_audio_atempo',
    'stretch_audio',
    'change_video_speed',
    'scale_srt_timestamps',
    'scale_ass_timestamps',
    'parse_ass_timestamp_to_ms',
    'ms_to_ass_timestamp',
    'get_default_output_path',
]