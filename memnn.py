import argparse
import collections
import copy
import glob

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions


def _encode(embed, sentences, length, position_encoding=False):
    xp = cuda.get_array_module(sentences)

    e = embed(sentences)
    if position_encoding:
        ndim = e.ndim
        n_batch, n_words, n_units = e.shape[:3]
        length = length.reshape(
            (n_batch,) + (1,) * (ndim - 1)).astype(numpy.float32)
        k = xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units
        # k = k.reshape((1,)*(ndim-2) +  (1, n_units))
        i = xp.arange(1, n_words + 1, dtype=numpy.float32)
        i = i.reshape((1,) * (ndim - 2) + (n_words, 1))
        coeff = (1 - i / length) - k * (1 - 2.0 * i / length)
        e = coeff * e
    s = F.sum(e, axis=-2)
    return s


class Memory(object):

    def __init__(self, A, C, TA, TC):
        self.A = A
        self.C = C
        self.TA = TA
        self.TC = TC

    def encode(self, sentence, lengths):
        mi = _encode(self.A, sentence, lengths)
        ci = _encode(self.C, sentence, lengths)
        return mi, ci

    def register_all(self, sentences, lengths=None):
        self.m, self.c = self.encode(sentences, lengths)

    def query(self, u):
        xp = cuda.get_array_module(u)
        m = self.m
        c = self.c
        batch, size = m.shape[:2]
        inds = xp.arange(size - 1, -1, -1, dtype=numpy.int32)
        tm = self.TA(inds)
        tc = self.TC(inds)
        tm = F.broadcast_to(tm, (batch,) + tm.shape)
        tc = F.broadcast_to(tc, (batch,) + tc.shape)
        p = F.softmax(F.batch_matmul(m + tm, u))
        o = F.batch_matmul(F.swapaxes(c + tc, 2, 1), p)
        o = o[:, :, 0]
        u = o + u
        return u


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab, max_memory=15):
        super(MemNN, self).__init__()
        normal = initializers.Normal()
        with self.init_scope():
            self.E1 = L.EmbedID(n_vocab, n_units, initialW=normal)
            self.E2 = L.EmbedID(n_vocab, n_units, initialW=normal)
            self.E3 = L.EmbedID(n_vocab, n_units, initialW=normal)
            self.E4 = L.EmbedID(n_vocab, n_units, initialW=normal)
            self.T1 = L.EmbedID(max_memory, n_units, initialW=normal)
            self.T2 = L.EmbedID(max_memory, n_units, initialW=normal)
            self.T3 = L.EmbedID(max_memory, n_units, initialW=normal)
            self.T4 = L.EmbedID(max_memory, n_units, initialW=normal)
            # self.B = L.EmbedID(n_vocab, n_units)
            # self.W = L.Linear(n_units, n_vocab)

        self.M1 = Memory(self.E1, self.E2, self.T1, self.T2)
        self.M2 = Memory(self.E2, self.E3, self.T2, self.T3)
        self.M3 = Memory(self.E3, self.E4, self.T3, self.T4)
        self.B = self.E1

    def fix_ignore_label(self):
        for embed in [self.E1, self.E2, self.E3, self.E4]:
            embed.W.data[0, :] = 0

    def register_all(self, sentences, lengths):
        self.M1.register_all(sentences, lengths)
        self.M2.register_all(sentences, lengths)
        self.M3.register_all(sentences, lengths)

    def query(self, question, lengths):
        u = _encode(self.B, question, lengths)
        u = self.M1.query(u)
        u = self.M2.query(u)
        u = self.M3.query(u)
        # a = self.W(u)
        a = F.linear(u, self.E4.W)
        return a

    def __call__(self, sentences, question):
        self.register_all(sentences, None)
        a = self.query(question, None)
        return a


def convert_data(train_data):
    data = []
    sentence_len = max(max(len(s.sentence) for s in story)
                       for story in train_data)
    for story in train_data:
        mem = numpy.zeros((50, sentence_len), dtype=numpy.int32)
        i = 0
        for sent in story:
            if isinstance(sent, data.Sentence):
                if i == 50:
                    mem[0:i - 1, :] = mem[1:i, :]
                    i -= 1
                mem[i, 0:len(sent.sentence)] = sent.sentence
                i += 1
            elif isinstance(sent, data.Query):
                query = numpy.zeros(sentence_len, dtype=numpy.int32)
                query[0:len(sent.sentence)] = sent.sentence
                data.append({
                    'sentences': mem.copy(),
                    'question': query,
                    'answer': numpy.array(sent.answer, 'i'),
                })

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chainer example: End-to-end memory networks')
    parser.add_argument('data', help='Path to bAbI dataset')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    args = parser.parse_args()

    import data
    vocab = collections.defaultdict(lambda: len(vocab))
    vocab['<unk>'] = 0

    for data_id in range(1, 21):

        train_data = data.read_data(
            vocab,
            glob.glob('%s/qa%d_*train.txt' % (args.data, data_id))[0])
        test_data = data.read_data(
            vocab,
            glob.glob('%s/qa%d_*test.txt' % (args.data, data_id))[0])
        print('Training data: %d' % len(train_data))

        train_data = convert_data(train_data)
        test_data = convert_data(test_data)

        memnn = MemNN(args.unit, len(vocab), 50)
        model = L.Classifier(memnn, label_key='answer')
        opt = optimizers.Adam()

        if args.gpu >= 0:
            chainer.cuda.get_device(args.gpu).use()
            model.to_gpu()

        opt.setup(model)

        train_iter = chainer.iterators.SerialIterator(
            train_data, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(
            test_data, args.batchsize, repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'))

        @training.make_extension()
        def fix_ignore_label(trainer):
            memnn.fix_ignore_label()

        trainer.extend(fix_ignore_label)
        trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()
