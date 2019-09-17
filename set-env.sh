#!/bin/bash

export ATT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export KALDI_ROOT=$ATT_DIR/deps/kaldi
export PYTHONPATH=$PYTHONPATH:$ATT_DIR/deps/kaldi-python/kaldi-python
export PYTHONPATH=$PYTHONPATH:$ATT_DIR/deps/kaldi/tools/openfst/lib/python2.7/site-packages
export PYTHONPATH=$PYTHONPATH:$ATT_DIR

. $ATT_DIR/exp/wsjs5/path.sh
