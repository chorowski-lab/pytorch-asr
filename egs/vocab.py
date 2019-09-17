import torchtext.vocab


class Vocab(object):
    def __init__(self, itos, stoi):
        self.itos = itos
        self.stoi = stoi

    @staticmethod
    def from_counter(counter):
        vocab = torchtext.vocab.Vocab(counter)
        return Vocab(vocab.itos, vocab.stoi)

    @staticmethod
    def from_file(fname):
        itos = []
        with open(fname, 'r') as f:
            for line in f:
                itos += [line[:-1]]
        stoi = {s: i for i, s in enumerate(itos)}
        return Vocab(itos, stoi)
