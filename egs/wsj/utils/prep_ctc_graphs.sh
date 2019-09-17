DIR=$1

WSJDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Generate the CTC decoding graph

#cat $WSJDIR/vocabulary.txt | sed -e 's/ $/<spc>/' > $DIR/net_vocab.txt
#python $WSJDIR/utils/build_ctc_graphs.py ctc_unigram \
#  $DIR/CTC.fst \
#  --net_vocab $DIR/net_vocab.txt \
#  --tokens $DIR/chars.txt \
#  --add_final_space
#fstcompose $DIR/CTC.fst $DIR/LG_syms.fst $DIR/CTC_LG_syms.fst

rm -Rf $DIR/biphone
mkdir -p $DIR/biphone

cp $DIR/{words.txt,chars.txt,net_vocab.txt} $DIR/biphone/
python $WSJDIR/utils/build_ctc_graphs.py ctc_bigram \
  $DIR/biphone/CTC_C.fst \
  --net_vocab $DIR/net_vocab.txt \
  --tokens $DIR/chars.txt \
  --add_final_space
fstcompose $DIR/biphone/CTC_C.fst $DIR/LG_syms.fst $DIR/biphone/CTC_LG_syms.fst
