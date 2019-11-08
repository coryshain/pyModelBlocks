#!/bin/bash

config_path="$1/config.ini";
pred_partition=$(basename $2 | rev | cut -d. -f2 | rev | cut -d'-' -f3);
cdr_dir=$(python3 -m mb.help -c CDRRepo_path | tr -d '\n');
export PYTHONPATH=$PYTHONPATH:$cdr_dir;
python3 -m cdr.bin.predict $config_path -p $pred_partition -e;
cat "$1/CDR/mse_losses_$pred_partition.txt";

