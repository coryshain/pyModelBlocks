#!/bin/bash

prdmeasures=$2;
resmeasures=$3;
mb/static_resources/scripts/predict-lmer.R $1 <(python mb/static_resources/scripts/merge_tables.py $prdmeasures $resmeasures subject docid sentid sentpos word)
