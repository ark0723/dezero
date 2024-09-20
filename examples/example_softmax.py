if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero.models import MLP
from dezero.optimizers import SGD, Adam
from dezero import Variable, as_variable
import dezero.functions as F
from dezero.datasets import Sprial
import matplotlib.pyplot as plt


dataset = Sprial(train=True)
data_size = len(dataset)
num_class = len(dataset.label_name)

# data visualization
dataset.show(grid=False)

# set parameters
epochs = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# np.ceil: 소수점 올림 / np.floor: 소수점 버림 / np.round: 소수점 반올림
iters = int(np.ceil(data_size / batch_size))

# loss list
loss_list = []

model = MLP((hidden_size, num_class))
optimizer = Adam(lr=lr).setup(model)

for epoch in range(epochs):
    # shuffle data
    dataset.shuffle()

    sum_loss = 0

    for i in range(iters):
        # mini batch
        batch_x, batch_t = dataset[i * batch_size : (i + 1) * batch_size]

        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)

        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # update loss per epoch
    total_loss = sum_loss / data_size
    loss_list.append(total_loss)
    print(f"epoch {epoch+1} : loss {total_loss:.2f}")

# accuracy
test_set = Sprial(train=False)
x, t = test_set.data, test_set.label
prob = model(x)
predicted = np.argmax(prob.data, axis=1)

accuracy = (predicted == t).mean()
print(f"accuracy: {accuracy:.4f}")


# loss graph
plt.plot(
    list(range(epochs)),
    loss_list,
    label="adam",
)

plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


# decision boundary using plt.contourf
grid_x = np.linspace(-1.0, 1.0, num=1000)
xx, yy = np.meshgrid(grid_x, grid_x)  # xx.shape = (1000, 1000)
# np.c_ : (Stack 1-D arrays as columns into a 2-D array)
x_in = np.c_[xx.ravel(), yy.ravel()]  # x_in.shape = (1000000, 2)
y_pred = np.argmax(model(x_in).data, axis=1).reshape(xx.shape)

plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
# c = t (color represent t: true label)
plt.scatter(x[:, 0], x[:, 1], c=t, s=40, cmap=plt.cm.cool)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
