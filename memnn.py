import copy
import glob
import random

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
import cupy

import nlputil


def _encode(embed, sentences, length, position_encoding=False):
    e = embed(chainer.Variable(sentences))
    if position_encoding:
        ndim = e.data.ndim
        n_batch = e.data.shape[0]
        n_words = e.data.shape[1]
        n_units = e.data.shape[2]
        length = length.reshape((n_batch,) + (1,) * (ndim - 1)).astype(numpy.float32)
        k = xp.arange(1, n_units + 1, dtype=numpy.float32) / n_units
        k = k.reshape((1,)*(ndim-2) +  (1, n_units))
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
        batch = m.data.shape[0]
        size = m.data.shape[1]
        inds, _ = xp.broadcast_arrays(
            xp.arange(size, dtype=numpy.int32)[::-1],
            xp.empty((batch, 1)))
        inds = chainer.Variable(inds)
        tm = self.TA(inds)
        tc = self.TC(inds)
        p = F.softmax(F.batch_matmul(m + tm, u))
        o = F.batch_matmul(F.swapaxes(c + tc, 2, 1), p)
        o = F.reshape(o, (batch, m.data.shape[2]))
        u = o + u
        return u


def init_params(*embs):
    for emb in embs:
        init_param(emb)


def init_param(emb):
    emb.W.data[:] = numpy.random.normal(0, 0.1, emb.W.data.shape)


class MemNN(chainer.Chain):

    def __init__(self, n_units, n_vocab, max_memory=15):
        super(MemNN, self).__init__(
            E1=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E2=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E3=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            E4=L.EmbedID(n_vocab, n_units),  # encoder for inputs
            T1=L.EmbedID(max_memory, n_units),  # encoder for inputs
            T2=L.EmbedID(max_memory, n_units),  # encoder for inputs
            T3=L.EmbedID(max_memory, n_units),  # encoder for inputs
            T4=L.EmbedID(max_memory, n_units),  # encoder for inputs
            #B=L.EmbedID(n_vocab, n_units),  # encoder for queries
            #W=L.Linear(n_units, n_vocab),
        )

        self.M1 = Memory(self.E1, self.E2, self.T1, self.T2)
        self.M2 = Memory(self.E2, self.E3, self.T2, self.T3)
        self.M3 = Memory(self.E3, self.E4, self.T3, self.T4)
        self.B = self.E1

        init_params(self.E1, self.E2, self.E3, self.E4,
                    self.T1, self.T2, self.T3, self.T4)

    def fix_ignore_label(self):
        for embed in [self.E1, self.E2, self.E3, self.E4]:
            embed.W.data[0, :] = 0

    def register(self, sentence, lengths):
        self.M1.register(sentence, lengths)
        self.M2.register(sentence, lengths)
        self.M3.register(sentence, lengths)

    def register_all(self, sentences, lengths):
        self.M1.register_all(sentences, lengths)
        self.M2.register_all(sentences, lengths)
        self.M3.register_all(sentences, lengths)

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

        accum_loss = None

        mem = xp.concatenate([mem.reshape((1, mem.shape[0],  mem.shape[1]))
                              for mem, _, _ in batch_data])
        query = xp.concatenate([query.reshape((1, query.shape[0]))
                                for _, query, _ in batch_data])
        answer = xp.array([answer for _, _, answer in batch_data], dtype=numpy.int32)

        model.register_all(mem, None)

        y = chainer.Variable(answer)
        loss, acc = model.query(query, None, y)

        if train:
            if accum_loss is None:
                accum_loss = loss
            else:
                accum_loss += loss

        total_loss += loss.data
        total_acc += acc.data
        count += 1

        if accum_loss is not None:
            model.zerograds()
            accum_loss.backward()
            opt.update()
            model.fix_ignore_label()

    #print('loss: %.4f\tacc: %.2f' % (float(total_loss), float(total_acc) / count * 100))
    return float(total_acc) / count


def convert_data(train_data, gpu):
    d = []
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
                if gpu >= 0:
                    d.append((chainer.cuda.to_gpu(mem),
                              chainer.cuda.to_gpu(query),
                              sent.answer))
                else:
                    d.append((copy.deepcopy(mem),
                              (query),
                              sent.answer))

    return d


if __name__ == '__main__':
    import data
    from chainer import optimizers
    vocab = nlputil.Vocabulary()
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
        train_data = convert_data(train_data, gpu)
        test_data = convert_data(test_data, gpu)

        model = MemNN(20, vocab.size, 50)
        opt = optimizers.Adam()
        #opt = optimizers.SGD(lr=0.01)
        #opt.add_hook(chainer.optimizer.GradientClipping(40))
        batch_size = 100
        numpy.seterr('raise')
        if gpu >= 0:
            model.to_gpu()
            xp = cupy
        else:
            xp = numpy
        opt.setup(model)

        for epoch in range(1000):
            if isinstance(opt, optimizers.SGD) and epoch % 25 == 24:
                opt.lr *= 0.5
            #print(epoch)

            random.shuffle(train_data)
            proc(train_data, batch_size, train=True)
            acc = proc(test_data, batch_size, train=False)

        acc = acc * 100
        err = 100 - acc
        print('%d: acc: %.2f\terr: %.2f' % (data_id, acc, err))
