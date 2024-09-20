if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero.optimizers import Adam
from dezero.models import SimpleLSTM
from dezero import SeqDataLoader
import dezero.datasets
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt
from dezero.utils import plot_dot_graph

max_epoch = 100
batch_size = 30
hidden_size = 100
lr = 0.001
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
# sequentail dataloader
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
data_size = len(train_set)

model = SimpleLSTM(hidden_size, out_size=1)
optimizer = Adam(lr=lr).setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mse(y, t)
        count += 1
        if count % bptt_length == 0 or count == data_size:
            # plot_dot_graph(loss)
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
    avg_loss = float(loss.data) / count
    print(f"| epoch {epoch + 1} | loss {avg_loss:4f}")

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
