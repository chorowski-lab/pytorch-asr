#!/bin/bash

set -e

D="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# We develop LMs more often that change the data.
#  - `exp` might link to local copy of the data (changes rarely)
#  - `exp_lm` might link to analogous dir shared over the network (changes often)
ln -s exp exp_lm

mkdir -p deps
cd $D

echo "Will: build your own Kaldi stack."
# Kaldi
cd $D/deps
git clone https://github.com/kaldi-asr/kaldi.git
git checkout ecd48ca7f9b9af116d4cdb5bcd65116311ff518c
cd $D/deps/kaldi/tools
OPENFST_CONFIGURE="--enable-static --enable-shared --enable-far --enable-ngram-fsts" make
extras/install_irstlm.sh
cd $D/deps/kaldi/src
./configure --shared --cudatk-dir=$CUDA_DIR
make depend -j 8
make -j 8

# Kaldi python wrappers
cd $D/deps
git clone https://github.com/janchorowski/kaldi-python.git
cd $D/deps/kaldi-python
KALDI_ROOT=$D/deps/kaldi make

mkdir -p exp/wsjs5
cd exp/wsjs5/
ln -s $KALDI_ROOT/egs/wsj/s5/* .
cat run.sh | sed -e 's:wsj0=/export/corpora5/LDC/LDC93S6B:wsj0=/pio/data/data/wsj/WSJ0:' -e 's:wsj1=/export/corpora5/LDC/LDC94S13B:wsj1=/pio/data/data/wsj/WSJ1:' > ./run_pio.sh

echo -e "\e[5m\e[31medit wile path in exp/wsjs5/run_pio.sh\nedit cmd.sh i path.sh\e[0m"
# KenLM
cd $D
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build/
#export BOOST_INCLUDEDIR=/net/archive/groups/plggneurony/os/anaconda2/include/
export BOOST_INCLUDEDIR=`dirname \`which python\``/../include
cmake ..
make -j 2

