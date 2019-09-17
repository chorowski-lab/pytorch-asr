import os
import re
import sys
from collections import defaultdict

import tabulate

DEV = 'dev93'
TEST = 'eval92'


top_dir = sys.argv[1]
tgt_step = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
drop_prefix = int(sys.argv[3]) if len(sys.argv) > 3 else None
mode = sys.argv[4] if len(sys.argv) > 4 else None
assert mode in ('nolm', 'lex', '') or mode is None
split = sys.argv[5] if len(sys.argv) > 5 else 'devtest'
assert split in ('test', 'devtest')

if split == 'test':
    DEV, TEST = TEST, DEV


def get_wer(fpath):
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith('%WER'):
                return float(line.split()[1])
    return None


results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for (exp, sub, files) in os.walk(sys.argv[1], followlinks=True):
    for f in files:
        if re.match(r'wer_\d+', f) is None:
            continue
        if mode == 'nolm' and ('chars' not in exp or 'lex' in exp):
            continue
        elif mode == 'lex' and 'lex' not in exp:
            continue
        elif not mode and 'chars' in exp:
            continue
        werfile = os.path.join(exp, f)
        wer = get_wer(werfile)
        acwt = int(f.split('_')[1])
        checkpoint = int(re.sub(r'.*checkpoint_(\d+).*', r'\1', exp))
        subset = exp.split('/')[-1]
        subset = DEV if DEV in subset else TEST
        expname = exp.split('/')[-2]
        expname = expname[drop_prefix:] if drop_prefix else expname
        results[expname][subset][checkpoint][acwt] = wer

dev_acwt = {}
rows = []
for expname in sorted(results.keys()):
    if DEV not in results[expname]:
        continue
    r = results[expname][DEV]
    checkpoints = r.keys()
    chpt = sorted(checkpoints, key=lambda c: abs(tgt_step - c))[0]
    (acwt, wer) = sorted(r[chpt].items(), key=lambda t: t[1])[0]
    acwt_range = '%d - %d' % (min(r[chpt].keys()), max(r[chpt].keys()))
    rows.append([wer, acwt, DEV, chpt, expname, acwt_range])
    dev_acwt[expname] = (chpt, acwt)

headers = ['%WER', 'acwt', 'set', 'step', 'expname', 'acwt range']
print tabulate.tabulate(rows, headers=headers)
rows = []

if split == 'test':
    sys.exit(0)

for expname in sorted(dev_acwt.keys()):
    if TEST not in results[expname]:
        continue
    r = results[expname][TEST]
    checkpoints = r.keys()
    chpt, acwt = dev_acwt[expname]
    wer = '---'
    acwt_range = '---'
    if chpt not in r:
        chpt = '---'
        acwt = '---'
    elif acwt not in r[chpt]:
        acwt = '---'
    else:
        acwt_range = '%d - %d' % (min(r[chpt].keys()), max(r[chpt].keys()))
        wer = r[chpt][acwt]
    rows.append([wer, acwt, TEST, chpt, expname, acwt_range])

print ''
print tabulate.tabulate(rows, headers=headers)
