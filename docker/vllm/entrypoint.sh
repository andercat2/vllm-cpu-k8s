#!/bin/sh
# Starts vLLM from the model the initContainer put on the volume.
set -eu

model_path="${MODEL_ROOT:?}/${MODEL_DIR:?}"

if [ ! -f "${model_path}/.download-complete" ]; then
    echo "Model is not ready at ${model_path}: the download marker is missing" >&2
    exit 1
fi

# These are set here and not in the ConfigMap so that a settings change cannot
# switch the access control or the privacy logging off by accident. The last
# middleware is the outermost, so rejected requests are audited too.
export VLLM_LOGGING_CONFIG_PATH=/opt/vllm-privacy/logging.json
unset VLLM_DEBUG_LOG_API_SERVER_RESPONSE

exec vllm serve "${model_path}" --config "${SERVE_CONFIG_FILE:?}" \
    --middleware vllm_privacy.middleware.RequireTokenMiddleware \
    --middleware vllm_privacy.middleware.AuditLogMiddleware \
    "$@"
