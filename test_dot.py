import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check

import dot


class TestDot(unittest.TestCase):
    shape = (3, 4, 5)

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (self.shape[0], self.shape[2])) \
                             .astype(numpy.float32)
        self.w = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.grad = numpy.random.uniform(-1, 1,
                                         (self.shape[0], self.shape[1])) \
                                .astype(numpy.float32)

    def check_forward(self, x_data, w_data):
        x = chainer.Variable(x_data)
        w = chainer.Variable(w_data)
        y = cuda.to_cpu(dot.dot(x, w).data)
        for b in range(w_data.shape[0]):
            for i in range(w_data.shape[1]):
                self.assertAlmostEqual(y[b, i],
                                       self.w[b, i].dot(self.x[b]),
                                       places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.w)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x),
                           cuda.to_gpu(self.w))

    def check_backward(self, x_data, w_data, grad):
        x = chainer.Variable(x_data)
        w = chainer.Variable(w_data)
        y = dot.dot(x, w)
        y.grad = grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data, w.data))
        gx, gw = gradient_check.numerical_grad(
            f, (x.data, w.data), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gw, w.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.w, self.grad)

    def test_backward_cpu(self):
        self.check_backward(cuda.to_gpu(self.x),
                            cuda.to_gpu(self.w),
                            cuda.to_gpu(self.grad))

