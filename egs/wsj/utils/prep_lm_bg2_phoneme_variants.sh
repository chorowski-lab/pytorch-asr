set -e
set -x



#DIR=$1
DIR=/home/jch/scratch/att_speech/exp/wsjs5/pydata/lm_bg_phn

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Generate the CTC decoding graph

make_graph () {
name=$1
shift
kind=$1
shift


rm -Rf $DIR/${name}
mkdir -p $DIR/${name}

cp $DIR/{words.txt,phonemes.txt} $DIR/${name}/
cat $WSJDIR/vocabulary_phn.txt  > $DIR/${name}/net_vocab.txt
python $WSJDIR/utils/build_ctc_graphs.py $kind \
  $DIR/${name}/CTC.fst \
  --net_vocab $DIR/${name}/net_vocab.txt \
  --tokens $DIR/${name}/phonemes.txt \
  "$@"
fstcompose $DIR/${name}/CTC.fst $DIR/LG_syms.fst $DIR/${name}/CTC_LG_syms.fst
}

make_graph monophone ctc_unigram
make_graph monophone_noloop ctc_unigram --no_allow_nonblank_selfloops

make_graph biphone ctc_bigram
make_graph biphone_noloop_contextblank ctc_bigram --no_allow_nonblank_selfloops --use_contextual_blanks
