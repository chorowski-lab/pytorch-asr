set -e
set -x

DIR=$1

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Generate the CTC decoding graph

rm -Rf $DIR/monophone
mkdir -p $DIR/monophone

cp $DIR/{words.txt,chars.txt} $DIR/monophone/
cat $WSJDIR/vocabulary.txt | sed -e 's/ $/<spc>/' > $DIR/monophone/net_vocab.txt
python $WSJDIR/utils/build_ctc_graphs.py ctc_unigram \
  $DIR/monophone/CTC.fst \
  --net_vocab $DIR/monophone/net_vocab.txt \
  --tokens $DIR/monophone/chars.txt \
  --add_final_space
fstcompose $DIR/monophone/CTC.fst $DIR/LG_syms.fst $DIR/monophone/CTC_LG_syms.fst

rm -Rf $DIR/biphone
mkdir -p $DIR/biphone

cp $DIR/monophone/{words.txt,chars.txt,net_vocab.txt} $DIR/biphone/
python $WSJDIR/utils/build_ctc_graphs.py ctc_bigram \
  $DIR/biphone/CTC_C.fst \
  --net_vocab $DIR/biphone/net_vocab.txt \
  --tokens $DIR/biphone/chars.txt \
  --add_final_space
fstcompose $DIR/biphone/CTC_C.fst $DIR/LG_syms.fst $DIR/biphone/CTC_LG_syms.fst
