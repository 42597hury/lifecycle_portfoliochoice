#!/bin/bash
# watchdog.sh - sync saved_runs to S3 then auto-terminate self
# usage: watchdog.sh <sweep_tmux_name> <instance_id> <api_key> [grace_sec] [max_wall_sec]
SWEEP_SESSION="$1"
INSTANCE_ID="$2"
API_KEY="$3"
GRACE_SEC=${4:-2700}      # 45 min grace after sweep ends (gives last cells time to save)
MAX_WALL_SEC=${5:-21600}  # 6 h hard cap on watchdog lifetime
S3_BUCKET=${S3_BUCKET:-hugo-thesis-runs}
AWS_REGION=${AWS_REGION:-eu-north-1}

start=$(date +%s)
log() { echo "[watchdog $(date '+%F %T')] $*" >> ~/watchdog.log; }
log "started: session=$SWEEP_SESSION instance=$INSTANCE_ID grace=${GRACE_SEC}s max_wall=${MAX_WALL_SEC}s s3=s3://${S3_BUCKET}/"

reason="unknown"
while true; do
    now=$(date +%s)
    elapsed=$((now - start))

    if [ $elapsed -ge $MAX_WALL_SEC ]; then
        reason="HARD_TIMEOUT after ${elapsed}s"
        log "$reason - skipping grace, syncing then terminating"
        break
    fi

    if ! tmux has-session -t "$SWEEP_SESSION" 2>/dev/null; then
        reason="sweep tmux session gone after ${elapsed}s"
        log "$reason - sleeping ${GRACE_SEC}s grace before sync+terminate"
        sleep $GRACE_SEC
        break
    fi

    sleep 30
done

# Periodic sync during the sweep would be nice, but we sync ONCE here at the end
# to keep the watchdog simple. Bundles get pushed to S3 and persist past terminate.
if [ -d ~/thesisscripts_JAX/saved_runs ]; then
    log "syncing ~/thesisscripts_JAX/saved_runs/ to s3://${S3_BUCKET}/saved_runs/ ..."
    SYNC_OUT=$(aws s3 sync ~/thesisscripts_JAX/saved_runs/ s3://${S3_BUCKET}/saved_runs/ \
        --region ${AWS_REGION} 2>&1)
    log "sync done: $(echo "$SYNC_OUT" | wc -l) lines"
    log "sync tail: $(echo "$SYNC_OUT" | tail -3 | tr '\n' ' | ')"
else
    log "no saved_runs dir found, skipping S3 sync"
fi

log "calling Lambda terminate API for $INSTANCE_ID (reason: $reason)..."
RESPONSE=$(curl -s -u "${API_KEY}:" \
    https://cloud.lambda.ai/api/v1/instance-operations/terminate \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\": [\"${INSTANCE_ID}\"]}")
log "terminate response: $RESPONSE"
