if "__file__" in globals():
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import dezero
from dezero.datasets import MNIST
from dezero.dataloaders import DataLoader
from dezero.models import CNN
from dezero.optimizers import Adam, SGD
from dezero.functions import softmax_cross_entropy, accuracy, relu
from dezero.visualization import CurveGraph

# set parameter
max_epoch = 5
batch_size = 100

train_set = MNIST(train=True)
test_set = MNIST(train=False)


num_class = len(train_set.labels().keys())
train_size = len(train_set)
test_size = len(test_set)

# dataset visualization
# train_set.show()

# dataloader
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = CNN()
optimizer = Adam().setup(model)

# load saved parameters
tmp_dir = os.path.join(os.path.expanduser("~"), "./Desktop/Dezero/result")
npz_dir = os.path.join(tmp_dir, "mnist_cnn.npz")
if os.path.exists(npz_dir):
    model.load_weights(npz_dir)

# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

# save loss and accuracy
train_loss, test_loss = [], []
train_acc, test_acc = [], []

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    # train
    for x, t in train_loader:
        x = x.reshape(-1, 1, 28, 28)
        y = model(x)
        loss = softmax_cross_entropy(y, t)
        acc = accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    total_loss, total_acc = sum_loss / train_size, sum_acc / train_size
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    print(
        f"epoch {epoch + 1} : train loss {total_loss :.4f} / train acc {total_acc:.4f}"
    )

    # test
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            x = x.reshape(-1, 1, 28, 28)
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    total_loss, total_acc = sum_loss / test_size, sum_acc / test_size
    test_loss.append(total_loss)
    test_acc.append(total_acc)

    print(f"epoch {epoch + 1} : test loss {total_loss :.4f} / test acc {total_acc:.4f}")

# save parameters
model.save_weights(npz_dir)

loss_g = CurveGraph(max_epoch, y_label="loss")
loss_g.set_legend("train", "test")
loss_g.show_graph(train_loss, test_loss, to_file="mnist_loss.png")

acc_g = CurveGraph(max_epoch, y_label="accuracy")
acc_g.set_legend("train", "test")
acc_g.show_graph(train_acc, test_acc, to_file="mnist_acc.png")
