#!/bin/bash
# init-qdrant.sh - Waits for Qdrant and restores snapshot if collection doesn't exist

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION="${QDRANT_COLLECTION:-dnd_rule_assistant}"
SNAPSHOT_DIR="/qdrant/snapshots/${COLLECTION}"

echo "Waiting for Qdrant to be ready..."
until curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" > /dev/null 2>&1; do
    sleep 1
done
echo "Qdrant is ready."

# Check if collection exists
COLLECTIONS=$(curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/collections")
if echo "$COLLECTIONS" | grep -q "\"name\":\"${COLLECTION}\""; then
    echo "Collection '${COLLECTION}' already exists. Skipping restore."
    exit 0
fi

# Find snapshot file
SNAPSHOT_FILE=$(ls -t ${SNAPSHOT_DIR}/*.snapshot 2>/dev/null | head -1)
if [ -z "$SNAPSHOT_FILE" ]; then
    echo "No snapshot found in ${SNAPSHOT_DIR}. Collection will need to be created manually."
    exit 0
fi

SNAPSHOT_NAME=$(basename "$SNAPSHOT_FILE")
echo "Restoring snapshot: ${SNAPSHOT_NAME}"

# Restore from snapshot
curl -X POST "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${COLLECTION}/snapshots/recover" \
    -H "Content-Type: application/json" \
    -d "{\"location\": \"file://${SNAPSHOT_FILE}\"}"

echo ""
echo "Snapshot restoration complete."
