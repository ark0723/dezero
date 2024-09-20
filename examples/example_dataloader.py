if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero.datasets import Sprial
from dezero import DataLoader
from dezero.models import MLP
from dezero.optimizers import Adam, SGD
import dezero.functions as F
import dezero
from dezero.visualization import CurveGraph

# set parameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# load dataset
train_set = Sprial(train=True)
test_set = Sprial(train=False)
num_class = len(train_set.label_name)
train_size = len(train_set)
test_size = len(test_set)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, num_class))
optimizer = Adam(lr=lr).setup(model)
train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []


for epoch in range(max_epoch):
    # train
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:  # train mini batch data
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss = sum_loss / train_size
    train_loss_list.append(train_loss)
    train_acc = sum_acc / train_size
    train_acc_list.append(train_acc)

    print(
        f"epoch {epoch + 1} : train loss {train_loss :.4f} / accuracy {train_acc:.4f}"
    )

    # test
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss = sum_loss / test_size
    test_loss_list.append(test_loss)
    test_acc = sum_acc / test_size
    test_acc_list.append(test_acc)

    print(f"epoch {epoch + 1} : test loss {test_loss :.4f} / accuracy {test_acc:.4f}")


loss_g = CurveGraph(max_epoch)
loss_g.set_legend("train", "test")
loss_g.show_graph(train_loss_list, test_loss_list, to_file="loss.png")

acc_g = CurveGraph(max_epoch, y_label="accuracy")
acc_g.set_legend("train", "test")
acc_g.show_graph(train_acc_list, test_acc_list, to_file="accuracy.png")
