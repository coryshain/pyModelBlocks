get_base () 
{
basename $1 | rev | cut -d. -f2- | rev
}

get_suffix ()

{
echo "$1" | rev | cut -d. -f1 | rev
}

config_path="$1/config.ini"

pred_partition=$(get_base $2);
pred_partition=$(get_suffix $pred_partition);
pred_partition="${pred_partition/_part/}"
pred_partition=$(echo $pred_partition | sed 's/fit/train/g' | sed 's/expl/dev/g' | sed 's/held/text/g')

cdr_dir=$(python3 -m mb.help -c CDRRepo_path | tr -d '\n')

export PYTHONPATH=$PYTHONPATH:cdr_dir

python3 -m dtsr.bin.predict $config_path -p $pred_partition

cat "$1_outdir/DTSR/mse_losses_$pred_partition.txt"

