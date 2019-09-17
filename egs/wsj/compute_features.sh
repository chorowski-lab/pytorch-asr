#!/bin/bash
# This script writes data to a new directory suitable for reading with
# the pytorch datasets

set -e
# set -o xtrace

datasets=(train_si284 train_si84 dev_dt_05 dev_dt_20 test_dev93 test_eval92 test_eval93)
main_train_set=${datasets[0]}

echo "Normalizing with respect to: $main_train_set" >&2

kaldi_exp_dir=exp/wsjs5

kaldi_exp_dir=`readlink -e $kaldi_exp_dir`
dir=${kaldi_exp_dir}/pydata/fbank80
feat_cmd='compute-fbank-feats --use-energy=true --num-mel-bins=80 '`
    `'IN ark:- | add-deltas ark:- OUT'

mkdir -p $dir

dir=`readlink -e $dir`

mkdir -p $dir/all

echo "Computed with $feat_cmd" >> $dir/README

cd $kaldi_exp_dir

for dt in ${datasets[*]}
do
    cat data/$dt/wav.scp
done | sort | uniq > $dir/all/wav.scp

interpolated_feat_cmd=`echo $feat_cmd | \
    sed -e "s;IN;scp:$dir/all/wav.scp;" | \
    sed -e "s;OUT;ark,scp:$dir/all/feats.ark,$dir/all/feats.scp;"`

echo "Running $interpolated_feat_cmd"
eval $interpolated_feat_cmd

for dt in ${datasets[*]}
do
    mkdir -p $dir/$dt
    cp data/$dt/text $dir/$dt/text
    cp data/$dt/utt2spk $dir/$dt/utt2spk
    utils/filter_scp.pl $dir/$dt/text $dir/all/feats.scp > $dir/$dt/feats.scp
    utils/filter_scp.pl $dir/$dt/text $dir/all/wav.scp > $dir/$dt/wav.scp
done

compute-cmvn-stats scp:$dir/$main_train_set/feats.scp $dir/$main_train_set/cmvn

python $ATT_DIR/egs/wsj/utils/apply_kaldi_lexicon.py egs/wsj/vocabulary_phn.txt \
  $kaldi_exp_dir/data/local/dict_larger/lexicon.txt $dir/*/text
