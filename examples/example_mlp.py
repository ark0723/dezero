# -> 해결방법은 현재 실행중인 파일 디렉토리의 부모 디렉토리(..)를 모듈 검색 경로에 추가한다.
# globals(): dict of global variables and symbols of current program
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
from dezero.models import MLP
from dezero.optimizers import SGD, Momentum, AdaGrad, RMSProp, AdaDelta, Adam
import matplotlib.pyplot as plt


# set parameters/hyper parmeters
lr = 0.2
iters = 10000

# generate dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

model = MLP((10, 1))
optimizer = Adam(lr=lr)
optimizer.setup(model)

# learning
for i in range(iters):
    y_pred = model(x)
    loss = F.mse(y, y_pred)

    # backpropagation
    model.cleargrads()
    loss.backward()

    # gradient decent
    optimizer.update()

    if i % 1000 == 0:
        print(loss)

# visualization of prediction
fig, ax = plt.subplots(figsize=(9, 9))
# original data
ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

# predicted y
x = np.sort(x, axis=0)  # sorting
ax.plot(x, model(x).data, color="b", lw=2.5)
plt.show()
