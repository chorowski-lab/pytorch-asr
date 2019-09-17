# -*- coding: utf8 -*-
import pickle
import os
import torch
from att_speech.configuration import Configuration
from att_speech.utils import do_evaluate
from att_speech import fst_utils

LM_FILE='/pio/scratch/2/jch/att_speech/exp/wsjs5/pydata/lm_ees_tg_larger/LG_syms.fst'
CONFIG_FILE='/pio/scratch/2/i248100/tcn_abest.yaml'
MODEL_FILE='/pio/scratch/2/i248100/tcn_abest.pkl'
BEAM_SIZE = 10

config = Configuration(CONFIG_FILE, {
    'Datasets.test.batch_size': '1',
    'Model.decoder.lm_file': LM_FILE
})
state_dict = torch.load(MODEL_FILE, map_location='cpu')

model = config['Model']
model.load_state(state_dict['state_dict'])
model = model.cuda()

dataset = config['Datasets']['test']

model.decoder.beam_size = BEAM_SIZE
model = model.eval()

def get_item(num, true_num=True):
    it = iter(dataset)
    for i in xrange(num):
        next(it)
    batch = next(it)

    feature_lens = batch['features'][1]
    features = batch['features'][0].cuda()

    text_lens = batch['texts'][1]
    texts = batch['texts'][0]
    return {
        'features': features,
        'feature_lens': feature_lens,
        'speakers': batch['spkids'],
        'texts': texts,
        'text_lens': text_lens
    }

def get_i(uttid):
    with open('/pio/scratch/2/i248100/att_speech/exp/wsjs5/pydata/fbank80/test_eval92/text') as f:
        for i, line in enumerate(f):
            _id = line.strip().split()[0]
            if _id == uttid:
                return i
    return -1

ordering = ['441c040f','443c0409','442c0403','440c040j','447c040g','446c040y','441c040c','447c040i','443c040w','441c040v','445c040m','440c040t','446c0403','443c040h','440c040g','442c040y','445c040p','443c0415','444c040l','442c040b','440c0412','441c040s','443c040j','446c0414','446c0410','442c040a','443c040s','444c040i','441c0401','444c040v','442c040i','446c0406','446c040c','444c0412','440c040k','447c0402','445c040z','442c0401','446c0415','447c040y','444c040x','441c040b','440c040l','441c040t','443c040k','444c040c','443c040v','447c040s','442c0411','444c0413','441c0408','440c0405','447c040k','441c0402','442c0412','443c0413','445c040t','445c0408','443c0403','442c0404','441c0403','441c040r','447c040w','444c040b','441c040h','441c040w','440c0404','443c040q','443c040y','446c040r','443c040z','440c040u','444c040p','447c040f','440c040n','441c040d','442c040f','447c040b','441c0413','444c040k','441c0407','441c0404','442c040o','440c040v','443c040m','446c040z','441c040e','442c040g','445c0411','441c040y','442c0405','447c040d','440c0413','444c040m','443c0408','440c0414','440c0406','443c040i','444c0408','446c040f','443c0406','441c040k','441c040z','447c040l','440c040h','444c0416','440c040a','443c040d','446c040o','445c0410','444c040n','447c040u','445c0414','447c040t','447c0411','441c0406','446c040p','446c040x','443c040o','442c0402','445c040r','442c040u','444c040q','442c040e','445c0407','443c0418','441c040l','441c0411','440c040o','445c0413','445c040d','444c040t','446c040k','447c0412','446c040h','441c0410','442c040q','440c0411','440c040z','445c040c','442c040c','442c0413','443c040e','441c0414','444c040o','440c040m','447c040e','443c0410','445c040q','444c0415','443c0412','445c040s','447c0414','447c0407','446c0408','445c040h','441c040x','443c0407','442c0408','441c0412','442c040p','440c0408','443c040g','442c0409','443c040x','444c040h','441c040i','440c0401','446c0411','440c040y','447c040v','443c0401','440c040s','447c040c','444c040a','443c0416','445c040y','446c040i','445c0404','444c0402','446c0401','444c0404','442c0410','443c0411','441c0415','442c0415','445c040v','445c040e','447c0409','447c040x','445c040a','444c0407','440c0417','447c040n','443c0414','445c040w','441c040n','445c040k','440c040d','444c0406','446c040a','447c040h','446c0409','446c040j','447c0405','441c040m','445c040j','447c0403','446c0405','443c0405','445c040x','444c040u','447c040r','446c040g','442c040r','440c0415','440c040c','443c040a','441c0409','445c0402','445c0405','440c0409','446c0402','442c040v','444c040r','440c040i','441c040p','441c040a','441c0405','443c0404','446c040e','445c040n','447c0410','446c040s','444c0401','445c0401','444c040z','445c040o','444c040w','445c040b','444c0405','446c0407','440c0403','442c0407','443c0417','442c040z','440c040q','442c040m','444c040e','447c0413','443c040b','442c040j','440c0407','447c040p','447c0401','446c040d','442c040x','440c040p','442c040l','444c040d','442c0406','443c040p','446c0404','444c0414','447c0408','440c040r','442c040n','444c040f','446c040b','444c0410','446c040u','445c0409','442c040t','447c040a','440c0410','442c040k','442c040w','444c040s','443c040n','445c040g','442c040h','445c040u','446c040n','446c040m','446c0413','443c040r','445c0406','445c040f','440c040e','443c040t','440c040f','441c040o','445c040l','447c040j','440c0402','442c040s','443c040u','446c0412','443c040c','447c040o','441c040u','443c040l','444c0403','445c0403','447c0415','441c040q','447c040q','447c0406','446c040v','444c040j','447c040m','443c040f','440c040x','442c0414','440c040b','446c040t','446c040q','443c0402','440c040w','447c0404','445c040i','440c0416','442c040d','446c040l','444c0409','444c040y','441c040j','444c040g','444c0411','445c0412','441c040g','446c040w','447c040z']

# Zdekodowanie z modelem językowym

model.decoder.lm_weight = 0.8
model.decoder.min_attention_pos = 0.3
model.decoder.coverage_tau = 0.1
model.decoder.coverage_weight = 0.5
model.decoder.length_normalization = 1.2
model.decoder.keep_eos_score = False
#model.decoder.debug = False
model.decoder.beam_size = BEAM_SIZE
model.decoder.use_graph_search = True
model.decoder.graph_search_history_len = -1 #1000

def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def rescore(sent):
    model.decoder.beam_size = 1
    sent = [dataset.dataset.vocabulary.stoi[c] for c in sent]
    model.decoder.rescore = sent
    return model.decode(print_debug=False, **item)

def score_acoustic(item, sent):
    sent = [dataset.dataset.vocabulary.stoi[c] for c in sent]
    encoded = model.encoder(item['features'], item['feature_lens'], item['speakers'], None)
    batch_size = encoded[0].size()[1]
    enc_state = model.decoder.enc_initial_state(encoded[0], encoded[1], 1, 1)
    my_logits = []
    sent = sent + [len(dataset.dataset.vocabulary.itos)]
    coverage = enc_state['att_weights'].detach()
    for i in range(len(sent)):
        prev_inputs = enc_state['inputs']
        logits, enc_state = model.decoder.enc_step(**enc_state)
        coverage += enc_state['att_weights'].detach()
        logprobs = torch.nn.functional.log_softmax(logits[0][0])
        my_logits += [ logprobs[sent[i]].item() ]
        new_input = torch.cuda.LongTensor([sent[i]])
        enc_state['inputs'] = torch.cat(
            (prev_inputs[1:], model.decoder.embedding(new_input).unsqueeze(0)))
        
    coverage_scores = (
        (coverage > model.decoder.coverage_tau).sum(dim=0).float() / coverage.size(0))
            # print(coverage_scores)
    coverage = model.decoder.coverage_weight*(
                torch
                .log(coverage_scores))
    return sum(my_logits), coverage.item()

def score_lm(sent):
    sent = [dataset.dataset.vocabulary.stoi[c] for c in sent]
    nodes = {model.decoder.lm.start(): 0}
    for s in sent + [49]: # +eos
        nodes = fst_utils.expand(model.decoder.lm, nodes, model.decoder.alphabet_mapping[s], use_log_probs=True)
    return fst_utils.reduce_weights(nodes.values(), True) * -0.8

def rescore2(item, sent):
    model.decoder.beam_size = 1
    acoustic, coverage = score_acoustic(item, sent)
    lm = score_lm(sent)
    return {
        'acoustic': acoustic,
        'coverage': coverage,
        'lm': lm,
        'loss': (acoustic + coverage + lm) / (len(sent) ** model.decoder.length_normalization)
    }


def dfs(graph):
    V = {g[0]: ((dataset.dataset.vocabulary.itos[g[1]] if g[1] != '<sos>' else ''),) + g[2:] for g in graph['V']}
    E = {g[0]: [] for g in graph['V']}
    for edg in graph['E']:
        E[edg[0]] += [edg[1]]
    subsents = {
        'sents': []
    }
    def do_dfs(node, cur_est):
        if V[node][3]:
            subsents['sents'] += [cur_est]
        for edg in E[node]:
            do_dfs(edg, cur_est + V[edg][0])
    do_dfs(graph['V'][0][0], '')
    return subsents['sents']

makedir('lattices3')
for i in xrange(len(dataset)):
    print("Processing sentence {}/{}".format(i+1, len(dataset)))
    i = get_i(ordering[i])
    if os.path.exists('lattices3/{}.graph'.format(i)):
        continue
    # if i != 6:
    #     continue
    item = get_item(i)
    text = item['texts'][0]
    toks = map(lambda x: dataset.dataset.vocabulary.itos[x], text)
    orig_toks = ''.join(toks)
    print("GROUNDTRUTH: {}".format(orig_toks))
    toks = map((lambda x: x if x != ' ' else '<spc>'), toks)

    model.decoder.rescore = None
    model.decoder.beam_size = BEAM_SIZE
    model.decoder.use_graph_search = True
    decoded = model.decode(print_debug=False, **item)

    decoded_sent = ''.join(map(lambda x: dataset.dataset.vocabulary.itos[x], decoded['decoded'][0]))
    print("DECODED: {}".format(decoded_sent))

    graph = decoded['graph'][0]
    sentences = dfs(graph)

    with open('lattices3/{}.graph'.format(i), 'wb') as f:
        pickle.dump({
            'graph': graph,
            'sentences': sentences}
            , f)

    makedir('lattices3/{}'.format(i))
    # best_loss = -10000
    # best_loss_i = -1

    # best_wer = 10000
    # best_wer_i = 0


    for j, sent in enumerate(sentences):
        print("Rescoring {}/{}".format(j+1, len(sentences)))
        print(sent)
        rescored = rescore2(item, sent)
        with open('lattices3/{}/{}.score'.format(i, j), 'wb') as f:
            pickle.dump(rescored, f)
        if len(sentences) > 5000:
            break
