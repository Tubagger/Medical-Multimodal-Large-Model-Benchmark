#!/usr/bin/env python3
"""Validate P.3 text, image jobs, and all existing generated images."""
import argparse
from pathlib import Path
from generate_qwen_images import DEFAULT_DATA, validate_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?', default=DEFAULT_DATA)
    validate_images(parser.parse_args().dataset.resolve())
