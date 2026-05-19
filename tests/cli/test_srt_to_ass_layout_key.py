import pytest
from utils.ass_utils import extract_layout_key_from_srt_text, convert_srt_segments_to_ass_dialogues

class TestLayer1_SrtLayoutKeyParser:
    def test_first_line_matching_layout_key_is_extracted(self):
        text = "top_left\nbody line 1\nbody line 2"
        key, body, warn = extract_layout_key_from_srt_text(
            text, 
            known_layout_keys={"top_left"}, 
            fallback_layout_key="default"
        )
        assert key == "top_left"
        assert body == "body line 1\nbody line 2"
        assert warn is None

    def test_single_line_matching_layout_key_is_body_text(self):
        text = "top_left"
        key, body, warn = extract_layout_key_from_srt_text(
            text, 
            known_layout_keys={"top_left"}, 
            fallback_layout_key="default"
        )
        assert key == "default"
        assert body == "top_left"
        assert warn is None

    def test_first_line_not_layout_warn_mode_uses_fallback(self):
        text = "random_text\nbody line 1"
        key, body, warn = extract_layout_key_from_srt_text(
            text, 
            known_layout_keys={"top_left"}, 
            fallback_layout_key="default", 
            mode="warn"
        )
        assert key == "default"
        assert body == "random_text\nbody line 1"
        assert warn is not None
        assert "Unknown layout key" in warn

    def test_first_line_not_layout_strict_mode_raises(self):
        text = "random_text\nbody line 1"
        with pytest.raises(ValueError, match="Unknown layout key"):
            extract_layout_key_from_srt_text(
                text, 
                known_layout_keys={"top_left"}, 
                fallback_layout_key="default", 
                mode="strict"
            )

    def test_escape_layout_key_prefix_backslash(self):
        text = "\\top_left\nbody line 1"
        key, body, warn = extract_layout_key_from_srt_text(
            text, 
            known_layout_keys={"top_left"}, 
            fallback_layout_key="default"
        )
        assert key == "default"
        assert body == "top_left\nbody line 1"
        assert warn is None

    def test_layout_key_with_empty_body_is_skipped(self):
        text = "top_left\n   \n\t"
        key, body, warn = extract_layout_key_from_srt_text(
            text, 
            known_layout_keys={"top_left"}, 
            fallback_layout_key="default"
        )
        assert key == "top_left"
        assert body == ""
        assert warn is not None
        assert "empty body" in warn

    def test_mode_off_keeps_first_line_as_body_text(self):
        text = "top_left\nbody line 1"
        key, body, warn = extract_layout_key_from_srt_text(
            text,
            known_layout_keys={"top_left"},
            fallback_layout_key="default",
            mode="off",
        )
        assert key == "default"
        assert body == text
        assert warn is None

    def test_invalid_mode_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid srt layout key mode"):
            extract_layout_key_from_srt_text(
                "top_left\nbody",
                known_layout_keys={"top_left"},
                fallback_layout_key="default",
                mode="bad",
            )

    def test_convert_srt_to_ass_writes_layout_key_to_name_field(self):
        segments = [{"startraw": "00:00:01,000", "endraw": "00:00:02,000", "text": "top_left\nHello"}]
        dialogues = convert_srt_segments_to_ass_dialogues(
            segments,
            known_layout_keys={"top_left"},
            layout_key="default"
        )
        assert len(dialogues) == 1
        assert "NoteStyle,top_left," in dialogues[0]
