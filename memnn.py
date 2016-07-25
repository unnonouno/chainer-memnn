import collections
import copy
import glob
import random

import numpy

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions
import cupy


def _encode(embed, sentences, length, position_encoding=False):
    e = embed(sentences)
    if position_encoding:
        ndim = e.data.ndim
        n_batch, n_words, n_units = e.data.shape[:3]
        length = length.reshape((n_batch,) + (1,) * (ndim - 1)).astype(numpy.float32)
        k = xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units
        #k = k.reshape((1,)*(ndim-2) +  (1, n_units))
        i = xp.arange(1, n_words + 1, dtype=numpy.float32)
        i = i.reshape((1,)*(ndim-2) +  (n_words, 1))
        coeff = (1 - i / length) - k * (1 - 2.0 * i / length)
        e = chainer.Variable(coeff) * e
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
        m = self.m
        c = self.c
        batch, size = m.data.shape[:2]
        inds = chainer.Variable(xp.arange(size, dtype=numpy.int32)[::-1])
        tm = self.TA(inds)
        tc = self.TC(inds)
        tm = F.broadcast_to(tm, (batch,) + tm.data.shape)
        tc = F.broadcast_to(tc, (batch,) + tc.data.shape)
        p = F.softmax(F.batch_matmul(m + tm, u))
        o = F.batch_matmul(F.swapaxes(c + tc, 2, 1), p)
        o = o[:, :, 0]
        u = o + u
        return u


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab, max_memory=15):
        normal = initializers.Normal()
        super(MemNN, self).__init__(
            E1=L.EmbedID(n_vocab, n_units, initialW=normal),  # encoder for inputs
            E2=L.EmbedID(n_vocab, n_units, initialW=normal),  # encoder for inputs
            E3=L.EmbedID(n_vocab, n_units, initialW=normal),  # encoder for inputs
            E4=L.EmbedID(n_vocab, n_units, initialW=normal),  # encoder for inputs
            T1=L.EmbedID(max_memory, n_units, initialW=normal),  # encoder for inputs
            T2=L.EmbedID(max_memory, n_units, initialW=normal),  # encoder for inputs
            T3=L.EmbedID(max_memory, n_units, initialW=normal),  # encoder for inputs
            T4=L.EmbedID(max_memory, n_units, initialW=normal),  # encoder for inputs
            #B=L.EmbedID(n_vocab, n_units),  # encoder for queries
            #W=L.Linear(n_units, n_vocab),
        )

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
        #a = self.W(u)
        a = F.linear(u, self.E4.W)
        return a

    def __call__(self, sentences, question):
        self.register_all(sentences, None)
        a = self.query(question, None)
        return a


def convert_data(train_data):
    tuples = []
    sentence_len = max(max(len(s.sentence) for s in story) for story in train_data)
    for story in train_data:
        mem = numpy.zeros((50, sentence_len), dtype=numpy.int32)
        mem_length = numpy.zeros((50,), dtype=numpy.int32)
        i = 0
        for sent in story:
            if isinstance(sent, data.Sentence):
                if i == 50:
                    mem[0:i-1, :] = mem[1:i, :]
                    mem_length[0:i-1] = mem_length[1:i]
                    i -= 1
                mem[i, 0:len(sent.sentence)] = sent.sentence
                mem_length[i] = len(sent.sentence)
                i += 1
            elif isinstance(sent, data.Query):
                query = numpy.zeros(sentence_len, dtype=numpy.int32)
                query[0:len(sent.sentence)] = sent.sentence
                tuples.append((copy.deepcopy(mem),
                               (query),
                               numpy.array(sent.answer, 'i')))

    return tuples


if __name__ == '__main__':
    import data
    vocab = collections.defaultdict(lambda: len(vocab))
    data_dir = '/home/unno/qa/tasks_1-20_v1-2'
    data_type = 'en'
    for data_id in range(1, 21):

        train_data = data.read_data(
            vocab,
            glob.glob('%s/%s/qa%d_*train.txt' % (data_dir, data_type, data_id))[0])
        test_data = data.read_data(
            vocab,
            glob.glob('%s/%s/qa%d_*test.txt' % (data_dir, data_type, data_id))[0])
        print('Training data: %d' % len(train_data))

        gpu = 0
        train_data = convert_data(train_data)
        test_data = convert_data(test_data)

        memnn = MemNN(20, len(vocab), 50)
        model = L.Classifier(memnn)
        opt = optimizers.Adam()
        batch_size = 100

        if gpu >= 0:
            model.to_gpu()
            xp = cupy
        else:
            xp = numpy
        opt.setup(model)

        train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
        test_iter = chainer.iterators.SerialIterator(test_data, batch_size,
                                                     repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, opt, device=gpu)
        trainer = training.Trainer(updater, (100, 'epoch'))

        @training.make_extension()
        def fix_ignore_label(trainer):
            memnn.fix_ignore_label()

        trainer.extend(fix_ignore_label)
        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy']))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.run()
