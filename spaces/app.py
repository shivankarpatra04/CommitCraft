"""Hugging Face Space entry point. scripts/push_space.py puts space.py and config.py next to it."""

from space import build_demo

build_demo().launch()
