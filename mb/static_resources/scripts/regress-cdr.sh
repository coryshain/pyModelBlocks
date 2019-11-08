#!/bin/bash

preds=${*:5:$#};
cdr_dir=$(python3 -m mb.help -c CDRRepo_path | tr -d '\n');
export PYTHONPATH=$PYTHONPATH:$cdr_dir;
cat $3 | python3 mb/static_resources/scripts/config_cdr.py $1 $2 $4 $preds;
python3 -m cdr.bin.train $4/config.ini;

