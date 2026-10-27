#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vendor
manifest=vendor/mpi-operator-v0.8.2.yaml
if [[ ! -f "$manifest" ]]; then
  curl -fsSL https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.8.2/deploy/v2beta1/mpi-operator.yaml -o "$manifest"
fi
printf '%s  %s\n' a7eafe901bebde1410ae0448c2d1ada214a64b2686f6aa44e74eff978573033f "$manifest" | sha256sum -c -
if ! kubectl get namespace mpi-operator >/dev/null 2>&1; then
  kubectl create namespace mpi-operator
fi
kubectl apply --server-side --dry-run=server -f "$manifest"
kubectl apply --server-side -f "$manifest"
kubectl -n mpi-operator rollout status deployment/mpi-operator --timeout=180s
