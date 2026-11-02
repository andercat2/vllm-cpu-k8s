#!/usr/bin/env bash
# Runs the load test against a temporary configuration of the deployment.
#
#   loadtest/run_experiment.sh NAME [--env KEY=VALUE]... [--arg FLAG]... [--scenario S]...
#
# --env changes the environment of the vllm container, --arg appends a flag to
# `vllm serve`, --scenario is passed to loadtest.py. The results are saved to
# loadtest/results/NAME.json. Undo everything with: kubectl apply -k k8s/overlays/minikube
set -euo pipefail
cd "$(dirname "$0")/.."

name=$1
shift

env_changes=()
serve_args=()
scenarios=()
while (($#)); do
    case "$1" in
        --env) env_changes+=("$2") ;;
        --arg) serve_args+=("$2") ;;
        --scenario) scenarios+=(--scenario "$2") ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift 2
done

args_json=$(python -c 'import json, sys; print(json.dumps(sys.argv[1:]))' ${serve_args[@]+"${serve_args[@]}"})
kubectl -n vllm patch deployment vllm --type=json \
    -p "[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/0/args\",\"value\":${args_json}}]"
if ((${#env_changes[@]})); then
    kubectl -n vllm set env deployment/vllm -c vllm "${env_changes[@]}"
fi
kubectl -n vllm rollout status deployment vllm --timeout=10m

kubectl -n vllm port-forward svc/vllm 18000:8000 >/dev/null 2>&1 &
forward_pid=$!
trap 'kill "${forward_pid}" 2>/dev/null || true' EXIT
until curl -s -m 2 -o /dev/null http://127.0.0.1:18000/health; do sleep 1; done

VLLM_API_KEY=$(grep '^VLLM_API_KEY=' k8s/overlays/minikube/secrets.env | cut -d= -f2 | tr -d '\r')
export VLLM_API_KEY
python -X utf8 loadtest/loadtest.py --base-url http://127.0.0.1:18000 --warmup --output "loadtest/results/${name}.json" \
    ${scenarios[@]+"${scenarios[@]}"}
