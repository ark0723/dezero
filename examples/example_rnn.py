if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero
from dezero.models import SimpleRNN
import matplotlib.pyplot as plt
from dezero.optimizers import Adam
import dezero.functions as F
import dezero.layers as L
from dezero.utils import plot_dot_graph

# dataset
train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

# draw graph
# xs = [data[0] for data in train_set]
# ts = [data[1] for data in train_set]

# plt.plot(np.arange(len(xs)), xs, label="xs")
# plt.plot(np.arange(len(ts)), ts, label="ts")
# plt.show()

# Hyperparameters
max_epoch = 100
hidden_size = 100
bptt_length = 30
lr = 0.001


model = SimpleRNN(hidden_size, 1)
optimizer = Adam(lr=lr).setup(model)

# training
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mse(y, t)
        count += 1

        # Truncated BPTT 타이밍 조정
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()  # disconnect creator
            optimizer.update()
    avg_loss = float(loss.data) / count
    print(f"epoch {epoch+1} | loss {avg_loss:4f}")

# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
plt.plot(np.arange(len(xs)), pred_list, label="predict")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
