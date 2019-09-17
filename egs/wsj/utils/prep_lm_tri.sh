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
  --trigram_to_class_file $ATT_DIR/ngrams/3grams.txt \
  --net_vocab $DIR/${name}/net_vocab.txt \
  --tokens $DIR/${name}/chars.txt \
  --add_final_space "$@"
fsttablecompose $DIR/${name}/CTC.fst $DIR/LG_syms.fst $DIR/${name}/CTC_LG_syms.fst
}

make_graph trichar ctc_trigram
