#!/usr/bin/env bash

ARTIFACT_DIR="$HOME/Downloads/artifact"
VERSION="$(awk -F"\"" '/version *= *\"/{print $2}' setup.py)"
echo "Version: $VERSION"
fd -HI -e whl "${VERSION}" -x cp {} -v "${ARTIFACT_DIR}"

PKG_DIR_NAME="Python-EasyGraph-${VERSION}-sdist-and-wheel-for-linux-and-intel-macos"

# in the same parent dir as artifact dir
PKG_DIR_PATH="$(dirname "${ARTIFACT_DIR}")/${PKG_DIR_NAME}"
mv -v "${ARTIFACT_DIR}" "${PKG_DIR_PATH}"
zip -r "${PKG_DIR_PATH}.zip" "${PKG_DIR_PATH}"

file.io-cli "${PKG_DIR_NAME}.zip"
