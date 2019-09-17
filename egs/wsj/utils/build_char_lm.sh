#!/bin/bash

set -e
set -x

KLM=$ATT_DIR/deps/kenlm/build/bin/
TRAINDIR=$1
DIR=$2

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

TMPDIR=$DIR/tmp_char_lm
mkdir -p $TMPDIR

cat $TRAINDIR/text | \
cut -d ' ' -f2- | \
sed -e 's/<NOISE>/~/g' -e "s/\`/\'/g" \
    -e 's/[^ ]* \(.*\)/\1/' \
    -e 's/ /$/g' -e 's/\(.\)/\1 /g' \
    -e 's/\$/<spc>/g' > $TMPDIR/lm_train.txt

$KLM/lmplz -o 2 < $TMPDIR/lm_train.txt > $TMPDIR/lm_bg.arpa

cat $TMPDIR/lm_bg.arpa | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst \
`#        --disambig-symbol='#0'` \
        --write-symbol-table=$TMPDIR/G_bg.syms \
        --keep-symbols \
        - | \
    fstprint | sed -e 's:<s>:<eps>:g' -e 's:</s>:<eps>:g' | \
    fstcompile \
         --isymbols=$TMPDIR/G_bg.syms \
         --osymbols=$TMPDIR/G_bg.syms \
         --keep_isymbols=true --keep_osymbols=true | \
    #fstdeterminizestar --use-log=true | \
    fstarcsort --sort_type=ilabel | \
    gzip -c \
    > $DIR/G_char_bg_syms.fst.gz

$KLM/lmplz -o 3 < $TMPDIR/lm_train.txt > $TMPDIR/lm_tg.arpa

cat $TMPDIR/lm_tg.arpa | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst \
`#        --disambig-symbol='#0'` \
        --write-symbol-table=$TMPDIR/G_tg.syms \
        --keep-symbols \
        - | \
    fstprint | sed -e 's:<s>:<eps>:g' -e 's:</s>:<eps>:g' | \
    fstcompile \
         --isymbols=$TMPDIR/G_tg.syms \
         --osymbols=$TMPDIR/G_tg.syms \
         --keep_isymbols=true --keep_osymbols=true | \
    fstarcsort --sort_type=ilabel | \
    gzip -c \
    > $DIR/G_char_tg_syms.fst.gz

#gzip -cd $DIR/G_char_tg_syms.fst.gz | \
#    fstdeterminizestar --use-log=true | \
#    fstminimize | \
#    gzip -c \
#    > $DIR/G_char_tg_syms_det.fst.gz
