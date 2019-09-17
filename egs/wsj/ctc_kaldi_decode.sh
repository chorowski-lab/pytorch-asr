#!/bin/bash

shopt -s nullglob  # Glob match "*" can produce an empty array

[[ $ATT_DIR == "" || $KALDI_ROOT == "" ]] && echo "ERROR: Source pio-env.sh first!" && exit 1

KU=$KALDI_ROOT/egs/wsj/s5/utils

subset=test
forward_opts="--polyak=0.9998 --no-strict"
min_acwt=0.5
max_acwt=1.0
model=""
pkl=""
lm_name=""
min_checkpoint=30000
max_checkpoint=1000000

source $KU/parse_options.sh

if [ $# -lt 1 ]; then
    echo "usage: `basename $0` <lm_dir> <model_dir> [<alignment_file>]"
    echo "Writes decoding results to decode_<subset>_<exp_name> dir."
    echo "options:"
    echo "        --use-bol (true|false)        #default: false, if true the graph will accout for bol symbol"
    echo "        --lm PATH                     #default: WSJ LM path (contains .fst and words.txt)"
    echo "        --min-acwt                    #default: 0.5, min acoustic weight"
    echo "        --max-acwt                    #default: 1.0, max acoustic weight"
    echo "        --subset                      #default: test"
    echo "        --pkl                         #default: .pkl filename w/o path (or ALL to decode every one)"
    echo "        --forward_opts                #default: none"
    echo "        --min-checkpoint              #default: 30000, skip if auto-selected checkpoint is from before 30k"
    echo "        --max-checkpoint              #default: 10**6, skip if auto-selected checkpoint is later"
    exit 1
fi

lm_dir=$1; shift
model_dir=$1; shift
ali_ws=${1:+ark,t:$1}; shift

if [ "$pkl" == "ALL" ]; then
    pushd $model_dir/checkpoints/
    pkl_list=(checkpoint_*.pkl)
    popd
elif [ "$pkl" != "" ]; then
    min_checkpoint=0
    pkl_list=("checkpoint_$pkl.pkl")
else
    min_checkpoint=0
    # Select the latest checkpoint
    pkl=$(ls ${model_dir}/checkpoints | grep -v logits | sort -t _ -k 2 -n | tail -1)
    pkl_list=("$pkl")
fi

echo "Using pkl $pkl"

# Select train yaml from the most recent run
yaml=$model_dir/$(ls -v ${model_dir} | grep -P 'train_config.*\.yaml' | tail -1)

[ "$yaml" == "" ] && echo "No yaml found!" && exit 1
[ "$pkl" == "" ] && echo "No .pkl selected!" && exit 1

for pkl in "${pkl_list[@]}" ; do

    pkl_path=${model_dir}/checkpoints/${pkl}
    [ ! -f "${pkl_path}" ] && echo "No .pkl checkpoint found at ${pkl_path}" && exit 1

    # Skip if early checkpoint
    checkpoint=$(python -c "import re, sys; print re.split(r'_|\.', sys.argv[1])[1]" $pkl)
    if [[ "$checkpoint" -lt "$min_checkpoint" || "$checkpoint" -gt $max_checkpoint ]]; then
        echo "Checkpoint step $checkpoint is < $min_checkpoint or > $max_checkpoint"
        echo "Skipping..."
        continue
    fi

    wsjdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    [ $subset = "test" ] && subset_name="eval92" || subset_name="dev93"
    opts_suffix=$(echo "$forward_opts" | sed 's/-/_/g' | sed s'/ /_/g' | sed s'/_\+/_/g')

    lm_name=$(basename $(dirname $lm_dir))"-"$(basename $lm_dir)
    dir=$model_dir/decode_${subset_name}_${lm_name}_$(basename $pkl ".pkl")"_"$opts_suffix
    tmp=$pkl_path"."$subset"_logits"  # dir/tmp_model_output.bin

    mkdir -p $dir || exit 1;
    mkdir -p $dir/scoring || exit 1;

    echo "Decoding time:  $(date)" | tee -ai $dir/info || exit 1; # Fail if cannot write
    echo "Config:         ${yaml}" | tee -ai $dir/info
    echo "Checkpoint:     $pkl_path" | tee -ai $dir/info
    echo "Output dir:     ${dir}" | tee -ai $dir/info
    echo "Alignment:      ${ali_ws}" | tee -ai $dir/info
    echo "" | tee -ai $dir/info

    # rm broken files
    find $dir -name "wer_*" -size 0 -exec rm {} \;
    find $dir -name "tra.*" -size 0 -exec rm {} \;

    # Check if all .wer_? files exist
    already_decoded=true
    for acwt in $(seq $max_acwt -0.1 $min_acwt); do
        acwt_int=$(python -c "print int($acwt * 10)")
        wer=$dir/wer_${acwt_int}
        if [ ! -f "$wer" ]; then
            already_decoded=false
            break
        fi
    done
    if [ $already_decoded = true ]; then
        echo "Already decoded in range ${min_acwt} -- ${max_acwt}."
        rm -f $tmp  # Remove old logits if exist
        continue
    fi

    if [ -f "$tmp" ]; then
        echo -e "    Logit file exists: $tmp\n"
    else
        # Our current Python scripts output all kinds of crap.
        # Better not pipe to decode-faster. Dump to hdd instead.
        python $ATT_DIR/ctc_forward.py --subset $subset $forward_opts \
                                       --model ${pkl_path} \
                                       $yaml ark,b:$tmp || exit 1;
    fi

    function decode_with_acwt () {
        acwt=$1; shift

        if [[ ! -f $lm_dir/CTC_LG_syms.fst ]]; then
            echo "Error: LM does not exist"
            exit 1
        fi

        echo -e "Decoding with acwt=${acwt}"
        acwt_int=$(python -c "print int($acwt * 10)")
        tra=$dir/scoring/tra.${acwt_int}
        wer=$dir/wer_${acwt_int}
        if [ -f "$wer" ]; then
            echo -e "    File exists: $wer\n"
            return
        fi

        fst=$lm_dir/CTC_LG_syms.fst
        symbols=$lm_dir/words.txt

        decode-faster \
            --acoustic-scale=$acwt \
            --word-symbol-table=$symbols \
            $fst ark,b:$tmp ark,t:$tra $ali_ws || exit 1;

        $KU/int2sym.pl -f 2- $symbols ${tra} > ${tra}.txt
        mv ${tra}.txt ${tra}

        compute-wer --text ark:$ATT_DIR/exp/wsjs5/pydata/fbank80/test_${subset_name}/text ark:$tra | \
            tee ${wer} || exit 1;
    }
    export -f decode_with_acwt
    export KU dir lm_dir tmp subset_name wsjdir ali_ws

    echo "lm_dir", $lm_dir

    for acwt in `seq $max_acwt -0.1 $min_acwt`; do
        echo $acwt
        decode_with_acwt $acwt
    done

    echo -e "\nDONE for ${pkl}"
    rm -f $tmp

done  # Loop over .pkl's
