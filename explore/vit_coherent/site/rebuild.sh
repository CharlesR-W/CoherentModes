#!/usr/bin/env bash
# rebuild viewer/site from data/vits_b2k (blocks are kept when unchanged)
cd "$(dirname "$0")/.." && uv run -q --with numpy --with pillow python site/build_site.py vits_b2k "$@" 2>&1 | grep -v "unchanged, kept"
