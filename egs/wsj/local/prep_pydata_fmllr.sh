#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
data_fmllr=pydata/fmllr_tri4b
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;


if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test
  for data in dev93 eval92; do
    dir=$data_fmllr/test_${data}
    steps/nnet/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
       --transform-dir exp/tri4b/decode_bd_tgpr_${data} \
       $dir data/test_${data} exp/tri4b $dir/log $dir/data || exit 1
  done
  # train
  dir=$data_fmllr/train_si284
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir exp/tri4b_ali_si284 \
     $dir data/train_si284 exp/tri4b $dir/log $dir/data || exit 1
fi

compute-cmvn-stats scp:${data_fmllr}/train_si284/feats.scp ${data_fmllr}/train_si284/cmvn

for data in train_si284 test_dev93 test_eval92; do
cp pydata/fbank80/$data/text ${data_fmllr}/$data
done
