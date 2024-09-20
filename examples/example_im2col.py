if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
from dezero import Variable
from dezero.utils import get_deconv_outsize

x1 = np.random.rand(1, 3, 7, 7)  # (N, C, H, W)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)

# conv2d simple version
N, C, H, W = 1, 3, 7, 7
OC, KH, KW = 8, 3, 3

x = Variable(np.random.rand(N, C, H, W))
weight = np.random.randn(OC, C, KH, KW)
y = F.conv2D_simple(x, weight, b=None, stride=1, pad=0)
y.backward()

print(y.shape)
print(x.grad.shape)


# tensordot
a = np.random.randint(2, size=(2, 3, 5))
b = np.random.randint(2, size=(3, 2, 4))
"""
<np.tensordot example>

1. case: one axis
a = np.random.randint(2, size=(2, 3, 5))
b = np.random.randint(2, size=(3, 2, 4))

c = np.tensordot(a, b, axes=((0), (1)))

explanation: 

a: (2, 3, 5) -> reduction of axis = 0
a: a collection of 2 matrices, each of shape = (3,5)
b: (3, 2, 5) -> reduction of axis =1
b: a collection of 3 matrices, each of shape = (2,5)

if # of matrices does not match: you will get 'shape-mismatch' error. 

2. case: two axis
a = np.random.randint(2, size=(2, 3, 4))
b = np.random.randint(2, size=(3, 4, 2))

# Performing tensordot
c = np.tensordot(a, b, axes=([1, 2], [0, 1])) -> # of matrices (3, 4) == (3, 4) 


"""

# x.shape = (N, c, h, w)
# W.shape = (oc, c, kh, kw)
# col.shape = (N, C, KH, KW, OH, OW)
col = np.random.rand(1, 3, 5, 5, 7, 7)
W = np.random.randn(4, 3, 5, 5)
b = np.random.randn(
    4,
)
y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))  # (N, OH, OW, OC)
y = np.rollaxis(y, 3, 1)  # (N, OC, OH, OW)
print(y.shape)  # 1, 4, 7, 7

x = np.random.randn(1, 3, 28, 28)
y = F.pooling_simple(x, kernel_size, stride=1, pad=0)
print(y.shape)


gy = np.random.rand(1, 3, 14, 14)
N, C, OH, OW = gy.shape
H, W = 28, 28
KH, KW = 2, 2


col = np.random.rand(N, C, OH, OW, KH, KW)
col = col.reshape(N, C, KH * KW, OH, OW)
idx = col.argmax(axis=2)
print(idx.shape)
indexes = idx.ravel() + np.arange(0, idx.size * KH * KW, KH * KW)
print(indexes.shape)
