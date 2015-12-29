import glob
import random

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
import cupy

import nlputil

import dot


def _encode(embed, sentences, length, position_encoding=True):
    e = embed(chainer.Variable(sentences))
    if position_encoding:
        n_batch = e.data.shape[0]
        n_words = e.data.shape[1]
        n_units = e.data.shape[2]
        length = length.reshape(n_batch, 1, 1).astype(numpy.float32)
        k = (xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units).reshape(1, 1, n_units)
        i = (xp.arange(1, n_words + 1, dtype=numpy.float32)).reshape(1, n_words, 1)
        coeff = (1 - i / length) - k * (1 - 2.0 * i / length)
        e = chainer.Variable(coeff) * e
    s = F.sum(e, axis=1)
    return s


class Memory(object):

    def __init__(self, A, C, TA, TC):
        self.A = A
        self.C = C
        self.TA = TA
        self.TC = TC
        self.ms = []
        self.cs = []

    def reset_state(self):
        self.ms = []
        self.cs = []

    def encode(self, sentence, lengths):
        mi = _encode(self.A, sentence, lengths)
        ci = _encode(self.C, sentence, lengths)
        return mi, ci

    def register(self, sentence, lengths):
        mi, ci = self.encode(sentence, lengths)
        mi = F.reshape(mi, (mi.data.shape[0], 1, mi.data.shape[1]))
        ci = F.reshape(ci, (ci.data.shape[0], 1, ci.data.shape[1]))
        self.ms.append(mi)
        self.cs.append(ci)

    def query(self, u):
        m = F.concat(self.ms)
        c = F.concat(self.cs)
        batch = m.data.shape[0]
        size = m.data.shape[1]
        inds, _ = xp.broadcast_arrays(
            xp.arange(size, dtype=numpy.int32)[::-1],
            xp.empty((batch, 1)))
        inds = chainer.Variable(inds)
        tm = self.TA(inds)
        tc = self.TC(inds)
        p = F.softmax(dot.dot(u, m + tm))
        o = dot.dot(p, F.swapaxes(c + tc, 2, 1))
        u = o + u
        return u


def init_params(*embs):
    for emb in embs:
        init_param(emb)


def init_param(emb):
    emb.W.data[:] = numpy.random.normal(0, 0.1, emb.W.data.shape)


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab):
        super(MemNN, self).__init__(
            E1=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E2=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E3=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E4=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            T1=L.EmbedID(15, n_units),  # encoder for inputs
            T2=L.EmbedID(15, n_units),  # encoder for inputs
            T3=L.EmbedID(15, n_units),  # encoder for inputs
            T4=L.EmbedID(15, n_units),  # encoder for inputs
            #B=L.EmbedID(n_vocab, n_units),  # encoder for queries
            #W=L.Linear(n_units, n_vocab),
        )

        self.M1 = Memory(self.E1, self.E2, self.T1, self.T2)
        self.M2 = Memory(self.E2, self.E3, self.T2, self.T3)
        self.M3 = Memory(self.E3, self.E4, self.T3, self.T4)
        self.B = self.E1

        init_params(self.E1, self.E2, self.E3, self.E4,
                    self.T1, self.T2, self.T3, self.T4)

    def reset_state(self):
        self.M1.reset_state()
        self.M2.reset_state()
        self.M3.reset_state()

    def register(self, sentence, lengths):
        self.M1.register(sentence, lengths)
        self.M2.register(sentence, lengths)
        self.M3.register(sentence, lengths)

    def query(self, question, lengths, y):
        u = _encode(self.B, question, lengths)
        u = self.M1.query(u)
        u = self.M2.query(u)
        u = self.M3.query(u)
        #a = self.W(u)
        a = F.linear(u, self.E4.W)
        return F.softmax_cross_entropy(a, y), F.accuracy(a, y)


def make_batch_sentence(lines):
    batch_size = len(lines)
    max_sent_len = max(len(line.sentence) for line in lines)
    # Fill zeros
    ws = numpy.zeros((batch_size, max_sent_len), dtype=numpy.int32)
    for i, line in enumerate(lines):
        ws[i, 0:len(line.sentence)] = line.sentence
    if xp is cupy:
        ws = chainer.cuda.to_gpu(ws)

    lengths = xp.array([len(line.sentence) for line in lines], dtype=numpy.int32)

    return ws, lengths


def proc(proc_data, batch_size, train=True):
    total_loss = 0
    total_acc = 0
    count = 0

    batch_size = min(batch_size, len(proc_data))

    for begin in range(0, len(proc_data), batch_size):
        end = min(len(proc_data), begin + batch_size)
        batch_data = proc_data[begin:end]
        max_len = max(len(d) for d in batch_data)
        model.reset_state()
        accum_loss = None
        for i in range(max_len):
            lines = [d[i] for d in batch_data]
            sentences, lengths = make_batch_sentence(lines)

            if all(isinstance(line, data.Sentence) for line in lines):
                model.register(sentences, lengths)
            elif all(isinstance(line, data.Query) for line in lines):
                y_data = xp.array([line.answer for line in lines], dtype=numpy.int32)
                y = chainer.Variable(y_data)
                loss, acc = model.query(sentences, lengths, y)

                if train:
                    if accum_loss is None:
                        accum_loss = loss
                    else:
                        accum_loss += loss

                total_loss += loss.data
                total_acc += acc.data
                count += 1
            else:
                raise

        if accum_loss is not None:
            model.zerograds()
            accum_loss.backward()
            opt.update()
            for link in model.links():
                if isinstance(link, L.EmbedID):
                    link.W.data[0, :] = 0

    print('loss: %.4f\tacc: %.2f' % (float(total_loss), float(total_acc) / count * 100))


if __name__ == '__main__':
    import data
    from chainer import optimizers
    vocab = nlputil.Vocabulary()
    data_dir = '/home/unno/qa/tasks_1-20_v1-2'
    data_type = 'en'
    data_id = 1

    train_data = data.read_data(
        vocab,
        glob.glob('%s/%s/qa%d_*train.txt' % (data_dir, data_type, data_id))[0])
    test_data = data.read_data(
        vocab,
        glob.glob('%s/%s/qa%d_*test.txt' % (data_dir, data_type, data_id))[0])
    print('Training data: %d' % len(train_data))

    model = MemNN(20, vocab.size)
    opt = optimizers.Adam()
    #opt = optimizers.SGD(lr=0.01)
    #opt.add_hook(chainer.optimizer.GradientClipping(40))
    batch_size = 500
    gpu = 0
    if gpu >= 0:
        model.to_gpu()
        xp = cupy
    else:
        xp = numpy
    opt.setup(model)

    for epoch in range(1000):
        if isinstance(opt, optimizers.SGD) and epoch % 25 == 24:
            opt.lr *= 0.5
        print(epoch)

        random.shuffle(train_data)
        proc(train_data, batch_size, train=True)
        proc(test_data, batch_size, train=False)
