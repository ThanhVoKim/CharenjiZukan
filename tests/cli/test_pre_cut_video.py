#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI parser tests for pre-cut remove/keep SRT modes."""

import pytest

from cli.pre_cut_video import build_parser


def test_keep_srt_mode_is_available():
    args = build_parser().parse_args([
        "--input", "source.mp4",
        "--output", "clips",
        "--keep-srt", "highlights.srt",
    ])

    assert args.keep_srt == "highlights.srt"
    assert args.remove_srt is None
    assert args.output == "clips"


def test_remove_and_keep_srt_are_mutually_exclusive():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--input", "source.mp4",
            "--output", "clean.mp4",
        ])

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--input", "source.mp4",
            "--output", "clean.mp4",
            "--remove-srt", "remove.srt",
            "--keep-srt", "highlights.srt",
        ])
