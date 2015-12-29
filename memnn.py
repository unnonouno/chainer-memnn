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


class Memory(chainer.Chain):

    def __init__(self, n_units, n_vocab):
        super(Memory, self).__init__(
            A=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            C=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            TA=L.EmbedID(15, n_units),
            TC=L.EmbedID(15, n_units),
        )
        self.A.W.data[:] = numpy.random.normal(0, 0.1, self.A.W.data.shape)
        self.C.W.data[:] = numpy.random.normal(0, 0.1, self.C.W.data.shape)
        self.TA.W.data[:] = numpy.random.normal(0, 0.1, self.TA.W.data.shape)
        self.TC.W.data[:] = numpy.random.normal(0, 0.1, self.TC.W.data.shape)
        self.m = None
        self.c = None

    def reset_state(self):
        self.m = None
        self.c = None

    def encode(self, sentence, ind):
        mi = _encode(self.A, sentence) + self.TA(ind)
        ci = _encode(self.C, sentence) + self.TC(ind)
        return mi, ci

    def register(self, sentence, ind):
        assert(isinstance(sentence, list))

        mi, ci = self.encode(sentence, ind)
        mi = F.reshape(mi, (mi.data.shape[0], 1, mi.data.shape[1]))
        ci = F.reshape(ci, (ci.data.shape[0], 1, ci.data.shape[1]))
        if self.m is None:
            self.m = mi
        else:
            self.m = F.concat([self.m, mi])

        if self.c is None:
            self.c = ci
        else:
            self.c = F.concat([self.c, ci])

    def query(self, u):
        p = F.softmax(dot.dot(u, self.m))
        o = dot.dot(p, F.swapaxes(self.c, 2, 1))
        u = o + u
        return u


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab):
        super(MemNN, self).__init__(
            M1=Memory(n_units, n_vocab),
            M2=Memory(n_units, n_vocab),
            M3=Memory(n_units, n_vocab),
            B=L.EmbedID(n_vocab, n_units),  # encoder for queries
            W=L.Linear(n_units, n_vocab),
        )
        self.B.W.data[:] = numpy.random.normal(0, 0.1, self.B.W.data.shape)
        self.W.W.data[:] = numpy.random.normal(0, 0.1, self.W.W.data.shape)

    def reset_state(self):
        self.M1.reset_state()
        self.M2.reset_state()
        self.M3.reset_state()

    def register(self, sentence, ind):
        self.M1.register(sentence, ind)
        self.M2.register(sentence, ind)
        self.M3.register(sentence, ind)

    def query(self, question, y):
        u = _encode(self.B, question)
        u = self.M1.query(u)
        u = self.M2.query(u)
        u = self.M3.query(u)
        a = self.W(u)
        #print('a: ', a.data.argmax(axis=1))
        #print('y: ', y.data)
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
                ind = xp.empty((batch_size,), numpy.int32)
                ind[:] = i
                model.register(sentences, chainer.Variable(ind))
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
