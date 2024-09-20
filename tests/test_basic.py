# test
import unittest
import numpy as np
from dezero import Variable


class OperationTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = x**3
        expected = Variable(np.array(8.0))
        self.assertEqual(y.data, expected.data)

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = x**3
        y.backward()  # implement Variable class's backward()
        expected = np.array(12.0)
        self.assertEqual(x.grad, expected)
