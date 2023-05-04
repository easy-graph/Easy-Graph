#!/usr/bin/env bash

ARTIFACT_DIR="$HOME/Downloads/artifact"
VERSION="$(awk -F"\"" '/version *= *\"/{print $2}' setup.py)"
echo "Version: $VERSION"
fd -HI -e whl "${VERSION}" -x cp {} -v "${ARTIFACT_DIR}"
