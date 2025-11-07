#!/usr/bin/env bash
set -e

MODEL_URL="https://github.com/yjymosheng/llm-rs/releases/download/v1.0.0/chat.zip"
TARGET_DIR="models/chat"
TMP_FILE="/tmp/chat_model.zip"

mkdir -p "$TARGET_DIR"

wget -O "$TMP_FILE" "$MODEL_URL"

unzip -o "$TMP_FILE" -d "$TARGET_DIR"

rm -f "$TMP_FILE"


