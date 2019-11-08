#!/bin/bash

model="$1.rdata";
prdmeasures=$2;
resmeasures=$3;
mb/static_resources/scripts/predict_lmer.R $model <(python ../resource-rt/scripts/merge_tables.py $prdmeasures $resmeasures subject docid sentid sentpos word)


