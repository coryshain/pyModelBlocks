#!/bin/bash

prdmeasures=$1;
resmeasures=$2;
bform=$3;
outfile=$4;
preds=${*:5:$#};

if [ -z "$preds" ]; then
	preds_STR="";
	preds_add_STR="";
	preds_ablate_STR="";
else
	until [ -z "$5" ]
	do
		if [ -z "$preds_STR" ]; then
			preds_STR=($4);
		else
			preds_STR+=($4);
		fi;
		if [[ $5 =~ ~.* ]]; then
			new=${5:1};
			if [ -z "$preds_ablate" ]; then
				preds_ablate=($new);
			else
				preds_ablate+=($new);
			fi;
		else
			new=$5;
		fi;
		if [ -z "$preds_add" ]; then
			preds_add=($new);
		else
			preds_add+=($new);
		fi;
		shift
	done
	preds_STR=$(IFS="_" ; echo "${preds_STR[*]}");
	preds_add_STR=$(IFS=\+ ; echo "${preds_add[*]}");
	preds_add_STR="-A $preds_add_STR";
	if [ ! -z "$preds_ablate" ]; then
		preds_ablate_STR=$(IFS=\+ ; echo "${preds_ablate[*]}");
		preds_ablate_STR="-a $preds_ablate_STR";
	else
		preds_ablate_STR="";
	fi;
fi;

corpusname=$(cut -d'.' -f1 <<< "$(basename $prdmeasures)");

mb/static_resources/scripts/regress-lmer.R <(python3 mb/static_resources/scripts/merge_tables.py $prdmeasures $resmeasures subject docid sentid sentpos word) $outfile -b $bform $preds_add_STR $preds_ablate_STR -c $corpusname -e;
