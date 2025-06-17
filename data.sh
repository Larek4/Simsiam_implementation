#!/bin/bash

# === User Configuration ===

# Set the maximum download limit (in GB)

LIMIT_GB=20

# Choose which modalities to download by setting them to true or false

MODALITY_S2L1C=true

MODALITY_S2L2A=true

MODALITY_S1GRD=false

MODALITY_S2RGB=false

# === Setup ===

# Convert limit from GB to bytes

LIMIT_BYTES=$((LIMIT_GB * 1024 * 1024 * 1024))

SPLIT="train"

BASE_DIR="data/ssl4eo-s12/$SPLIT"

TOTAL_SIZE=0

# === Function to Download Files for a Given Modality ===

download_modality() {

local MODALITY=$1

local URL_BASE="https://datapub.fz-juelich.de/ssl4eo-s12/${SPLIT}/${MODALITY}"

local TARGET_DIR="${BASE_DIR}/${MODALITY}"

echo "Downloading modality: $MODALITY from $URL_BASE"

mkdir -p "$TARGET_DIR"

cd "$TARGET_DIR" || exit 1

for i in $(seq -f "%06g" 1 10000); do

FILE="ssl4eos12_${SPLIT}_seasonal_data_${i}.zarr.zip"

URL="${URL_BASE}/${FILE}"

if wget -q --show-progress "$URL"; then

TOTAL_SIZE=$(du -sb "$BASE_DIR" | cut -f1)

TOTAL_MB=$((TOTAL_SIZE / 1024 / 1024))

echo "Total downloaded so far: ${TOTAL_MB} MB"

if [ "$TOTAL_SIZE" -ge "$LIMIT_BYTES" ]; then

echo "Reached the download limit of ${LIMIT_GB} GB. Stopping."

exit 0

fi

else

echo "File not found or download failed: $FILE"

break

fi

done

}

# === Main Execution ===

echo "Starting SSL4EO-S12 download for the '$SPLIT' split"

echo "Download limit: ${LIMIT_GB} GB"

# Trigger downloads for the selected modalities

if $MODALITY_S2L1C; then download_modality "S2L1C"; fi

if $MODALITY_S2L2A; then download_modality "S2L2A"; fi

if $MODALITY_S1GRD; then download_modality "S1GRD"; fi

if $MODALITY_S2RGB; then download_modality "S2RGB"; fi

echo "Download completed or limit reached."

