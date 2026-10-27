#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE=${IMAGE:-docker.io/library/easygraph-distributed:k8s-20261016}
buildah bud --layers --format docker --isolation chroot \
  --security-opt seccomp=unconfined --ignorefile deploy/k8s/Dockerfile.dockerignore \
  -t "$IMAGE" -f deploy/k8s/Dockerfile .
