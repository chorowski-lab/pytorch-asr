set -e
set -x



DIR=$1

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Generate the CTC decoding graph

make_graph () {
name=$1
shift
kind=$1
shift


rm -Rf $DIR/${name}
mkdir -p $DIR/${name}

cp $DIR/{words.txt,chars.txt} $DIR/${name}/
cat $WSJDIR/vocabulary.txt | sed -e 's/ $/<spc>/' > $DIR/${name}/net_vocab.txt
python $ATT_DIR/decoding_utils/ctc_fst.py $kind \
  $DIR/${name}/CTC.fst \
  --net_vocab $DIR/${name}/net_vocab.txt \
  --tokens $DIR/${name}/chars.txt \
  --add_final_space "$@"
fstcompose $DIR/${name}/CTC.fst $DIR/LG_syms.fst $DIR/${name}/CTC_LG_syms.fst
}

make_graph monophone ctc_unigram
make_graph monophone_noloop ctc_unigram --no_allow_nonblank_selfloops

make_graph biphone ctc_bigram
make_graph biphone_contextblank ctc_bigram --use_contextual_blanks
make_graph biphone_noloop_contextblank ctc_bigram --no_allow_nonblank_selfloops --use_contextual_blanks
make_graph biphone_looprepeats ctc_bigram --loop_using_symbol_repetitions
