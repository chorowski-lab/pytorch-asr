# pytorch-asr

Speech recognition models in PyTorch build on Kaldi data pre-processing toolchain.
Made at the University of Wrocław, Poland.

The repository holds the code to replicate the experiments from:
 * [Towards Using Context-Dependent Symbols in CTC Without State-Tying Decision Trees](https://arxiv.org/abs/1901.04379)
 * *Lattice Generation in Attention-based Speech Recognition Models*

# Setup

## Setting the Environment
1. Download the [`miniconda 3`](https://docs.conda.io/en/latest/miniconda.html)
   for your platform, then install it.
2. If you didn't add the conda to your `.bashrc` during installation, run the command provided by the installator, e.g.
   `eval "$(/pio/scratch/1/alan/miniconda2/bin/conda shell.bash hook)"`
   to populate your current shell with conda programs.
3. Install the `conda` environment by
   `conda env create -f environment.yml`

   To update the environment use `conda env update --file environment.yml`.

## Enabling the Environment
1. If you haven't done that already
   `eval "$(/pio/scratch/1/alan/miniconda2/bin/conda shell.bash hook)"`
2. Activate the appriopriate environment
    ```
    conda activate pytorch_asr
   source set-env.sh
    ```

## Dependencies

Run `build_deps.sh` script
    ```
    cd pytorch-asr
    ./build_deps.sh
    ```

# Training & Decoding

Model training
```
python train.py egs/wsj/deep_speech2.yaml /tmp/experiment
```
Greedy model decoding
```
python decode.py /tmp/experiment/config_train1.yaml --model /tmp/experiment/checkpoints/best.pkl
```
To decode using external LM, first build language models
```
bash egs/wsj/build_decoding_fst.sh
```
Then decode
```
bash egs/wsj/ctc_kaldi_decode.sh --min_checkpoint 35000 --pkl ALL --subset test lm/lm_ees_tg_larger/biphone runs/ctc_bi
```
Decoding results will be put in the experiment dir. Consult both scripts for more decoding options.

## Experiments: Towards Using Context-Dependent Symbols in CTC Without State-Tying Decision Trees

The models are located in `egs/wsj/yamls/*.yaml`.
To train a particular model
```
python train.py egs/wsj/yaml/ctcg_bi_cde.yaml runs/ctcg_bi_cde
```
Intermediate checkpoints and best model will be stored in `runs/ctcg_bi_cde/checkpoints`.
To decode, select the appriopriate language model.

| Model                             | $lm_path                                                  |
|-----------------------------------|-----------------------------------------------------------|
| mono-char CTC                     | exp/wsjs5/pydata/lm/lm_ees_tg_larger/monophone            |
| bi-char CTC, CTC-G, CTC-G (+ CDE) | exp/wsjs5/pydata/lm/lm_ees_tg_larger/biphone              |
| bi-char CTC-GB (+ CDE)            | exp/wsjs5/pydata/lm/lm_ees_tg_larger/biphone_contextblank |
```
bash egs/wsj/ctc_kaldi_decode.sh --min-acwt 0.3 --subset dev $lm_path runs/ctcg_bi_cde
```
Consult the decoding script for more options.

## Experiments: Lattice Generation in Attention-based Speech Recognition Models

First, train the initial model with CTC:
```
python train.py egs/wsj/yamls/lattice_decoding/ctc.yaml runs/lattice_base
```
The run the second stage with TCN
```
python train.py egs/wsj/yamls/lattice_decoding/ctc.yaml runs/lattice_stage2 --initialize-from runs/lattice_base/checkpoints/best.pkl
```
In order to decode, pick a checkpoint:
```
python decode.py ~/group/mza/recreate/tcn.yaml --model runs/lattice_stage2/checkpoints/best_51853_CER_0.0808282271662.pkl
--csv decoded.csv -m Model.decoder.use_graph_search True
Model.decoder.length_normalization 0 Model.decoder.coverage_weight 0.8
Model.decoder.min_attention_pos 0 Model.decoder.coverage_tau 0.25
Model.decoder.keep_eos_score False Model.decoder.lm_weight 0.75
Model.decoder.att_force_forward "[-10, 50]" Model.decoder.beam_size 10
Model.decoder.lm_file
/net/archive/groups/plggneurony/mza/lm_ees_tg_larger/LG_syms.fst
Datasets.test.batch_size 1
```
