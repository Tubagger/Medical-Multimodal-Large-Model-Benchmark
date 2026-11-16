#!/usr/bin/env python3
"""Validate S.3 frozen inputs without importing or calling models."""
import argparse
import json
from pathlib import Path
from build import DEFAULT_OUTPUT, validate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, nargs='?', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset.resolve()), ensure_ascii=False, indent=2))
