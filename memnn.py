import random

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
import cupy

import nlputil

import dot


def _encode(embed, sentence):
    vs = []
    for w in sentence:
        valid = (w.data != 0).reshape((w.data.shape[0], 1))
        e = embed(w)
        valid, _ = xp.broadcast_arrays(valid, e.data)
        valid = chainer.Variable(valid)
        zero = chainer.Variable(xp.zeros_like(e.data))
        e = F.where(valid, e, zero)
        e = F.reshape(e, (e.data.shape[0], 1, e.data.shape[1]))
        vs.append(e)
    v = F.concat(vs, axis=1)
    s = F.sum(v, axis=1)
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

    def encode(self, sentence):
        mi = _encode(self.A, sentence)
        ci = _encode(self.C, sentence)
        return mi, ci

    def register(self, sentence):
        assert(isinstance(sentence, list))

        mi, ci = self.encode(sentence)
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
        assert inds.shape == (batch, size)
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

    def register(self, sentence):
        self.M1.register(sentence)
        self.M2.register(sentence)
        self.M3.register(sentence)

    def query(self, question, y):
        u = _encode(self.B, question)
        u = self.M1.query(u)
        u = self.M2.query(u)
        u = self.M3.query(u)
        #a = self.W(u)
        a = F.linear(u, self.E4.W)
        return F.softmax_cross_entropy(a, y), F.accuracy(a, y)


def proc(proc_data, train=True):
    accum_loss = 0
    accum_acc = 0
    count = 0

    for begin in range(0, len(proc_data), batch_size):
        indexes = list(range(begin, begin + batch_size))
        max_len = max(len(proc_data[b]) for b in indexes)
        model.reset_state()
        for i in range(max_len):
            lines = []
            for b in indexes:
                d = proc_data[b]
                lines.append(d[i])
            max_sent_len = max(len(line.sentence) for line in lines)
            sentences = []
            for t in range(max_sent_len):
                ws = [line.sentence[t] if t < len(line.sentence) else 0 for line in lines]
                ws_data = xp.array(ws, dtype=numpy.int32)
                word = chainer.Variable(ws_data)
                sentences.append(word)

            if all(isinstance(line, data.Sentence) for line in lines):
                model.register(sentences)
            elif all(isinstance(line, data.Query) for line in lines):
                y_data = xp.array([line.answer for line in lines], dtype=numpy.int32)
                y = chainer.Variable(y_data)
                loss, acc = model.query(sentences, y)

                if train:
                    accum_loss += loss.data
                    accum_acc += acc.data
                    model.zerograds()
                    loss.backward()
                    opt.update()
                else:
                    accum_acc += acc.data
                count += 1
            else:
                raise

    print(accum_loss, accum_acc / count)


if __name__ == '__main__':
    import data
    from chainer import optimizers
    vocab = nlputil.Vocabulary()
    train_data = data.read_data(
        vocab,
        '/home/unno/qa/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt')
        #'/home/unno/qa/tasks_1-20_v1-2/en-10k/qa12_conjunction_train.txt')
    #'/home/unno/qa/tasks_1-20_v1-2/en-10k/qa20_agents-motivations_train.txt')
    test_data = data.read_data(
        vocab,
        '/home/unno/qa/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt')
    print('Training data: %d' % len(train_data))

    model = MemNN(50, vocab.size)
    opt = optimizers.Adam()
    #opt = optimizers.SGD()
    opt.setup(model)
    batch_size = len(train_data)
    model.to_gpu()
    xp = cupy

    for epoch in range(1000):
        if True:
            #indexes = list(range(batch_size))
            random.shuffle(train_data)
            proc(train_data, train=True)
            proc(test_data, train=False)

        else:
            for one_data in train_data:
                model.reset_state()
                for line in one_data:
                    wid = line.sentence
                    sentence = [chainer.Variable(numpy.array([w], dtype=numpy.int32))
                                for w in wid]
                    if isinstance(line, data.Sentence):
                        model.register(sentence)
                    else:
                        y = chainer.Variable(numpy.array([line.answer], dtype=numpy.int32))
                        loss, acc = model.query(sentence, y)

                        accum_loss += loss.data
                        model.zerograds()
                        loss.backward()
                        opt.update()
