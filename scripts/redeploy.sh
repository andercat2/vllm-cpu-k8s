#!/usr/bin/env bash
# Rebuilds both images, loads them into minikube and applies the manifests.
# Set MINIKUBE to use a binary that is not on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

minikube_bin=${MINIKUBE:-minikube}

docker build -f docker/model-downloader/Dockerfile -t vllm-model-downloader:1.0.0 .
docker build -f docker/vllm/Dockerfile -t vllm-cpu-private:0.32.0 .

# A running pod holds on to the old image, so stop it before replacing the tag.
kubectl -n vllm scale deployment vllm --replicas=0 2>/dev/null || true
kubectl -n vllm wait --for=delete pod -l app.kubernetes.io/name=vllm --timeout=120s 2>/dev/null || true
"${minikube_bin}" image rm vllm-cpu-private:0.32.0 vllm-model-downloader:1.0.0 2>/dev/null || true

"${minikube_bin}" image load vllm-model-downloader:1.0.0
"${minikube_bin}" image load vllm-cpu-private:0.32.0

kubectl apply -k k8s/overlays/minikube
kubectl -n vllm scale deployment vllm --replicas=1
kubectl -n vllm rollout status deployment vllm --timeout=15m
