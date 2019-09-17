#!/bin/bash

#
# Mostly taken from https://github.com/rizar/fully-neural-lvsr
# Originally under the MIT license
#

KU=$KALDI_ROOT/egs/wsj/s5/utils

use_bol=false

source $KU/parse_options.sh

if [ $# -ne 2 ]; then
    echo "usage: `basename $0` <lm_file> <dir>"
    echo "options:"
    echo "		--use-bol (true|false)        #default: false, if true the graph will accout for bol symbol"
    exit 1
fi

LMFILE=$1
DIR=$2

# LMFILE=$ATT_DIR/exp/wsjs5/data/local/nist_lm/lm_bg.arpa.gz
# DIR=$ATT_DIR/exp/wsjs5/pydata/lm_bg

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p $DIR

if [[ $LMFILE = *.gz ]]; then
    cat_cmd="gzip -d -c"
else
    cat_cmd="cat"
fi

#Get the word list from the LM
{
    echo "<eps>";
    $cat_cmd $LMFILE | \
    #skip up to \data\
    #then skip up to \1-grams
    #then print up to \2-grams or \end
    #then delete empty lines
        sed -e '0,/^\\data\\/d' \
            -e '0,/^\\1-grams:/d' \
            -e '/\(^\\2-grams:\)\|\(^\\end\\\)/,$d' \
            -e '/^\s*$/d' | \
    #print just the word
        awk '{print $2; }' | \
    #finally remove <s> and </s>, they will be added later
        grep -v '</\?s>'
    echo "#0";
    echo "<s>";
    echo "</s>";
} | awk '{ print $0, NR-1;}' > $DIR/words.txt

{
    echo "<eps>";
    echo "<spc>";
    cat $WSJDIR/vocabulary.txt | tr -d ' ' | sed -e '/^\s*$/d';
} | awk '{ print $0, NR-1;}' > $DIR/chars.txt


allowed_characters=`cat $DIR/chars.txt  | grep -v '<.*>' | awk '{print $1}' | tr -d ' \n'`
if echo $allowed_characters | grep '-' > /dev/null; then
    allowed_characters=`echo $allowed_characters | tr -d '-'`
    allowed_characters="${allowed_characters}-"
fi
echo "character set: $allowed_characters"

# Spell unknowns as noise, which we encode as ~
echo "<UNK> ~ <spc>" > $DIR/lexicon.txt

cat $DIR/words.txt | cut -d ' ' -f 1 | \
    grep -v "<.*>" | \
    grep -v "#0" > $DIR/tmp-words-to-convert.txt

cat $DIR/tmp-words-to-convert.txt | \
    tr -c -d "\n$allowed_characters" | sed -e "s/\(.\)/ \1/g" | \
    sed -e "s/$/ <spc>/" | \
    paste -d '' $DIR/tmp-words-to-convert.txt - >> $DIR/lexicon.txt

rm $DIR/tmp-words-to-convert.txt

$cat_cmd $LMFILE | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    $KU/eps2disambig.pl | $KU/s2eps.pl | \
    fstcompile \
        --isymbols=$DIR/words.txt \
        --osymbols=$DIR/words.txt \
        --keep_isymbols=false \
        --keep_osymbols=false | \
    fstrmepsilon | \
    fstarcsort --sort_type=ilabel      \
    > $DIR/G.fst

ndisambig=`$KU/add_lex_disambig.pl $DIR/lexicon.txt $DIR/lexicon_disambig.txt`

ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST. It won't hurt even if don't use it
( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$DIR/disambig.txt

cat $DIR/chars.txt | cut -d ' ' -f 1 | \
    #add disambiguation symbols
    cat - $DIR/disambig.txt | \
    awk '{ print $0, NR-1;}' > $DIR/chars_disambig.txt

$KU/make_lexicon_fst.pl \
    $DIR/lexicon_disambig.txt |\
    fstcompile \
        --isymbols=$DIR/chars_disambig.txt \
        --osymbols=$DIR/words.txt \
        --keep_isymbols=false --keep_osymbols=false |\
    fstaddselfloops  \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/chars_disambig.txt` |" \
        "echo `grep -oP '(?<=#0 )[0-9]+' $DIR/words.txt` |"  | \
    fstarcsort --sort_type=olabel > $DIR/L_disambig.fst

fsttablecompose $DIR/L_disambig.fst $DIR/G.fst | \
    fstdeterminizestar --use-log=true | \
    fstrmsymbols <(cat $DIR/chars_disambig.txt | grep '#' | cut -d ' ' -f 2) | \
    fstrmepslocal | \
    fstminimizeencoded | \
    fstarcsort --sort_type=ilabel | \
    cat	> $DIR/LG.fst

fstsymbols --isymbols=$DIR/chars.txt --osymbols=$DIR/words.txt --verify \
    $DIR/LG.fst $DIR/LG_syms.fst


# Generate the CTC decoding graph

python - << END
import os
import pywrapfst as fst

net_vocab = [s[:-1] for s in open('$WSJDIR/vocabulary.txt').readlines()]

CTC = fst.Fst(arc_type='standard')
CTC_os = fst.SymbolTable.read_text('$DIR/chars.txt')
CTC_is = fst.SymbolTable()
CTC_is.add_symbol('<eps>', 0)
for i, s in enumerate(net_vocab):
    CTC_is.add_symbol(s, i + 1)

CTC.set_input_symbols(CTC_is)
CTC.set_output_symbols(CTC_os)

after_blank = CTC.add_state()
CTC.set_start(after_blank)
CTC.set_final(after_blank)

l2s = {'<pad>': after_blank}

in2out_exceptions = {
    ' ': '<spc>'
}
in2out_label = lambda k: in2out_exceptions.get(k, k)

for i in range(CTC_is.num_symbols()):
    i = CTC_is.get_nth_key(i)
    l = CTC_is.find(i)
    if l in ('<pad>', '<eps>'):
        continue
    l2s[l] = CTC.add_state()
    CTC.set_final(l2s[l])

weight_one = fst.Weight.One('tropical')

final_space = CTC.add_state()
CTC.set_final(final_space)
final_space_arc = fst.Arc(CTC_is.find('<eps>'), CTC_os.find('<spc>'), weight_one, final_space)

os_eps = CTC_os.find('<eps>')

for l, s in l2s.items():
    in_label = CTC_is.find(l)
    out_label = os_eps if l == '<pad>' else CTC_os.find(in2out_label(l))

    # Self-loop, don't emit
    CTC.add_arc(s, fst.Arc(in_label, os_eps, weight_one, s))

    # Transition from another state - this emits
    for l2, s2 in l2s.items():
        if l == l2:
            continue
        CTC.add_arc(s2, fst.Arc(in_label, out_label, weight_one, s))

    # Optional transition to emit the final space
    CTC.add_arc(s, final_space_arc)

CTC.arcsort('olabel')

CTC.write('$DIR/CTC.fst')
END

fstcompose $DIR/CTC.fst $DIR/LG_syms.fst $DIR/CTC_LG_syms.fst
