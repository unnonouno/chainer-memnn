import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Dot(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, w_type = in_types

        type_check.expect(
            w_type.dtype == numpy.float32,
            x_type.dtype == numpy.float32,
            w_type.ndim == 3,
            x_type.ndim == 2,
            w_type.shape[0] == x_type.shape[0],
            w_type.shape[2] == x_type.shape[1],
        )

    def forward_cpu(self, inputs):
        x, w = inputs
        return numpy.einsum('ik,ijk->ij', x, w),

    def forward_gpu(self, inputs):
        x, w = inputs
        out = cuda.cupy.empty((w.shape[0], w.shape[1]),
                              dtype=numpy.float32)
        out = cuda.elementwise(
            'raw T x, raw T w, int32 n_out, int32 n_units',
            'raw T out',
            '''
            int batch = i / n_out;
            float v = 0;
            for (int j = 0; j < n_units; ++j) {
              v += x[batch * n_units + j] * w[i * n_units + j];
            }
            out[i] = v;
            ''',
            'dot_fwd'
        )(x, w, w.shape[1], w.shape[2], out, size=out.size)
        return out,

    def backward(self, inputs, grads):
        x, w = inputs
        g, = grads

        xp = cuda.get_array_module(*inputs)
        if xp is numpy:
            gx = numpy.einsum('ijk,ij->ik', w, g)
        else:
            gx = cuda.cupy.empty_like(x)
            cuda.elementwise(
                'raw T w, raw T g, int32 n_out, int32 n_units',
                'raw T gx',
                '''
                int batch = i / n_units;
                int k = i - batch * n_units;
                float v = 0;
                for (int j = 0; j < n_out; ++j) {
                  v += w[(batch * n_out + j) * n_units + k]
                           * g[batch * n_out + j];
                }
                gx[i] = v;
                ''',
                'dot_bwd'
            )(w, g, w.shape[1], w.shape[2], gx, size=gx.size)

        batch = x.shape[0]
        units = x.shape[1]
        out = g.shape[1]
        gw = x.reshape(batch, 1, units) * g.reshape(batch, out, 1)
        return gx, gw


def dot(x, w):
    return Dot()(x, w)
