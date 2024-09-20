if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero.datasets import Sprial
from dezero.transforms import Compose, Normalize, AsType

f = Compose([Normalize(mean=0.0, std=2.0), AsType(np.float64)])
train_set = Sprial(train=True, transform=f)
print(len(train_set))
batch_idx = [1, 2, 3]
batch_x, batch_t = train_set[batch_idx]

print(batch_x.shape, batch_t.shape)
