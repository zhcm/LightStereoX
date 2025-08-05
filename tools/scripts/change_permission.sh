#!/bin/bash

source tools/scripts/constant.sh

save_root_dir="${CKPT_ROOT_DIR%/}/$1"

useradd -u 1002 chenming.zhang
chown -R chenming.zhang:chenming.zhang $save_root_dir
