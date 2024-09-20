if "__file__" in globals():
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import test_mode
import dezero.functions as F
from dezero.utils import get_conv_outsize

x = np.ones(5)
print(x)

# train
y = F.dropout(x)
print(y)

# test
with test_mode():
    y = F.dropout(x)
    print(y)


# calculate conv outsize
H, W = 4, 4  # 입력 형상 (4x4)
KH, KW = 3, 3  # 필터/커널 사이즈 (3x3)
SH, SW = 1, 1  # stride size 1 (가로, 세로)
PH, PW = 1, 1  # padding size 1

output_H = get_conv_outsize(H, KH, SH, PH)
output_W = get_conv_outsize(W, KW, SW, PW)
print(output_H, output_W)
