if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero.models import VGG16
import dezero
from PIL import Image
import matplotlib.pyplot as plt
from dezero.utils import get_conv_outsize


model = VGG16(pretrain=True)

# x = np.random.rand(1, 3, 244, 244).astype(np.float32)
# model.plot(x)

img_dir = "C:/Users/ark07/Downloads/zebra.jpg"
img = Image.open(img_dir)
# img.show()

x = VGG16.preprocess(img)
# print(type(x), x.shape)  # <class 'numpy.ndarray'> (3, 224, 224)
x = x[np.newaxis]  # 배치용 축 추가

with dezero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)

model.plot(x)
labels = dezero.datasets.ImageNet.labels()
print(labels[predict_id])

print(get_conv_outsize(14, 3, 1, 1))
