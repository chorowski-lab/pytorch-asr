#!/bin/bash

set -e
set -x

#
# Mostly taken from https://github.com/rizar/fully-neural-lvsr
# Originally under the MIT license
#

KU=$KALDI_ROOT/egs/wsj/s5/utils

kaldi_exp_dir=$ATT_DIR/exp/wsjs5
kaldi_exp_dir=`readlink -e $kaldi_exp_dir`


use_bol=false
lexicon=$kaldi_exp_dir/data/local/dict_larger/lexicon.txt

source $KU/parse_options.sh

#if [ $# -ne 2 ]; then
#    echo "usage: `basename $0` <lm_file> <dir>"
#    echo "options:"
#    echo "		--use-bol (true|false)        #default: false, if true the graph will accout for bol symbol"
#    exit 1
#fi
#
#LMFILE=$1
#DIR=$2

LMFILE=$ATT_DIR/exp/wsjs5/data/local/nist_lm/lm_bg.arpa.gz
DIR=$ATT_DIR/exp/wsjs5/pydata/lm_bg_phn

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $DIR

if [[ $LMFILE = *.gz ]]; then
    cat_cmd="gzip -d -c"
else
    cat_cmd="cat"
fi

# cp $lexicon $DIR
paste -d ' ' <(cat $lexicon | cut -d ' ' -f 1) \
  <(cat $lexicon | cut -d ' ' -f 2- | tr -d '0-9') > $DIR/lexicon.txt


#Get the word list from the lexicon
{
    echo "<eps>";
    cat $DIR/lexicon.txt | cut -d ' ' -f 1;
    echo "#0";
    echo "<s>";
    echo "</s>";
} | awk '{ print $0, NR-1;}' > $DIR/words.txt


$cat_cmd $LMFILE | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    sed -e 's:<unk>:_:g' | \
    $KU/eps2disambig.pl | $KU/s2eps.pl | \
    fstcompile \
        --isymbols=$DIR/words.txt \
        --osymbols=$DIR/words.txt \
        --keep_isymbols=false \
        --keep_osymbols=false | \
    fstrmepsilon | \
    fstarcsort --sort_type=ilabel      \
    > $DIR/G.fst

{
    echo "<eps>";
    echo "<pad>";
    cat $DIR/lexicon.txt | cut -d ' ' -f 2-  | tr ' ' '\n' | sort | uniq
} | awk '{ print $0, NR-1;}' > $DIR/phonemes.txt


ndisambig=`$KU/add_lex_disambig.pl $DIR/lexicon.txt $DIR/lexicon_disambig.txt`

ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST. It won't hurt even if don't use it
( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$DIR/disambig.txt

cat $DIR/phonemes.txt | cut -d ' ' -f 1 | \
    #add disambiguation symbols
    cat - $DIR/disambig.txt | \
    awk '{ print $0, NR-1;}' > $DIR/phonemes_disambig.txt

$KU/make_lexicon_fst.pl \
    $DIR/lexicon_disambig.txt |\
    fstcompile \
        --isymbols=$DIR/phonemes_disambig.txt \
        --osymbols=$DIR/words.txt \
        --keep_isymbols=false --keep_osymbols=false |\
    fstaddselfloops  \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/phonemes_disambig.txt` |" \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/words.txt` |"  | \
    fstarcsort --sort_type=olabel > $DIR/L_disambig.fst

fsttablecompose $DIR/L_disambig.fst $DIR/G.fst | \
    fstdeterminizestar --use-log=true | \
    fstrmsymbols <(cat $DIR/phonemes_disambig.txt | grep '#' | cut -d ' ' -f 2) | \
    fstrmepslocal | \
    fstminimizeencoded | \
    fstarcsort --sort_type=ilabel | \
    cat	> $DIR/LG.fst

fstsymbols --isymbols=$DIR/phonemes.txt --osymbols=$DIR/words.txt --verify \
    $DIR/LG.fst $DIR/LG_syms.fst

