if "__file__" in globals():
    import sys, os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero.layers import Layer
from dezero.core import Parameter
import numpy as np

layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))


params_dict = {}
layer._flatten_params(params_dict)
print(params_dict)
