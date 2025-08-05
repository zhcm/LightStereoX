#!/bin/bash

source tools/scripts/constant.sh

current_datetime=$(date +"%Y%m%d-%H%M%S")
sourcepath="$PROJECT_ROOT_DIR"
targetpath="${WORKSPACE_DIR%/}/LightStereoX-$current_datetime"
if [ ! -d "$targetpath" ]; then
    cp -a "$sourcepath" "$targetpath" || { echo "Copy failed"; exit 1; }
else
    echo "Target path already exists: $targetpath"
fi

cd "$targetpath" || { echo "Failed to cd to $targetpath"; exit 1; }

source "${MINICONDA_DIR%/}/etc/profile.d/conda.sh"
conda activate stereo
export PYTHONPATH="./:$PYTHONPATH"
export HF_ENDPOINT='https://hf-mirror.com'
#export TORCH_HOME='/file_system/vepfs/algorithm/chenming.zhang/.cache/torch'
#export HF_HOME='/file_system/vepfs/algorithm/chenming.zhang/.cache/huggingface'