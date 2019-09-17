#!/bin/bash

bash $ATT_DIR/egs/wsj/build_decoding_fst.sh \
  /pio/scratch/1/alan_bis/att_speech/egs/wsj_make_lm/data/local/local_lm/3gram-mincount/lm_pr*.gz \
  $ATT_DIR/exp/wsjs5/pydata/lm_ees_tgpr_larger

bash $ATT_DIR/egs/wsj/build_decoding_fst.sh \
  /pio/scratch/1/alan_bis/att_speech/egs/wsj_make_lm/data/local/local_lm/3gram-mincount/lm_unpruned.gz \
  $ATT_DIR/exp/wsjs5/pydata/lm_ees_tg_larger

bash $ATT_DIR/egs/wsj/build_decoding_fst.sh \
  /pio/scratch/1/alan_bis/att_speech/egs/wsj_make_lm/data/local/local_lm/4gram-mincount/lm_pr*.gz \
  $ATT_DIR/exp/wsjs5/pydata/lm_ees_fgpr_larger

bash $ATT_DIR/egs/wsj/build_decoding_fst.sh \
  /pio/scratch/1/alan_bis/att_speech/egs/wsj_make_lm/data/local/local_lm/4gram-mincount/lm_unpruned.gz \
  $ATT_DIR/exp/wsjs5/pydata/lm_ees_fg_larger
