# Oryginalny model
Oryginalny model jest tutaj: /net/archive/groups/plggneurony/mza/tcn/experiments/tcn_params_check/temp_1.25_dil_2_hs_384_lpb_2

# Jak odtworzyć eksperymenty

0. Odtwarzane na branchu rnn_lattice
1. ./train-prometheus.sh ~/group/mza/recreate/ctc.yaml ~/group/mza/recreate/stage1/
2. ./train-prometheus.sh ~/group/mza/recreate/tcn.yaml ~/group/mza/recreate/stage2 --initialize-from ~/group/mza/recreate/stage1/checkpoints/best_189459_CER_0.0884155117819.pkl

** Można też użyć lepszego modelu bazowego. Jest tutaj: /net/people/plgjch/plggneurony/jch/att_speech/scratch_runs/wsj_debug_fst/mono_fst_ctc_R2/checkpoints/best_119289_CER_0.0573726761604.pkl

# Dekodowanie
python decode.py ~/group/mza/recreate/tcn.yaml --model ~/group/mza/recreate/stage2/checkpoints/best_51853_CER_0.0808282271662.pkl --csv decoded.csv -m Model.decoder.use_graph_search True Model.decoder.length_normalization 0 Model.decoder.coverage_weight 0.8 Model.decoder.min_attention_pos 0 Model.decoder.coverage_tau 0.25 Model.decoder.keep_eos_score False Model.decoder.lm_weight 0.75 Model.decoder.att_force_forward "[-10, 50]" Model.decoder.beam_size 10 Model.decoder.lm_file /net/archive/groups/plggneurony/mza/lm_ees_tg_larger/LG_syms.fst Datasets.test.batch_size 1

