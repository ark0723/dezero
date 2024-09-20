if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero.dataloaders import SeqDataLoader
from dezero.datasets import SinCurve

train_set = SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=3)
x, t = next(dataloader)
print(x)
print("=" * 10)
print(t)
