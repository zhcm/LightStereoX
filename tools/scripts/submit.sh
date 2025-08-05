#!/bin/bash

source tools/scripts/constant.sh

current_datetime=$(date +"%Y%m%d-%H%M%S")
sourcepath="$PROJECT_ROOT_DIR"
targetpath="${WORKSPACE_DIR%/}/LightStereoX-$current_datetime"
cp -a "$sourcepath" "$targetpath" || { echo "Copy failed"; exit 1; }

entrypoint='cd '"$targetpath"'
source '"${MINICONDA_DIR%/}"'/etc/profile.d/conda.sh
conda activate stereo
export PYTHONPATH="./:$PYTHONPATH"
bash tools/scripts/train.sh --cfg_file cfgs/nmrf/nmrf_swint_mix.py --extra_tag mix2_v2
bash tools/scripts/eval_stereoanything.sh output/MixDataset/NMRF/mix2_v2
bash tools/scripts/change_permission.sh output/MixDataset/NMRF/mix2_v2'


volc ml_task submit -c tools/scripts/job_conf_example.yaml --set Entrypoint="$entrypoint" --set TaskName=mix2_v2
