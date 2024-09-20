import numpy as np

x = np.array([1, 2, 3])
np.save("test.npy", x)

x = np.load("test.npy")
print(x)


# x1 = np.array([1, 2, 3])
# x2 = np.array([4, 5, 6])

# np.savez("test.npz", x1=x1, x2=x2)
# arrays = np.load("test.npz")
# x1 = arrays["x1"]
# x2 = arrays["x2"]
# print(x1)
# print(x2)


x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
data = {"x1": x1, "x2": x2}  # 키워드를 파이썬 딕셔너리로 묶음

np.savez_compressed("test.npz", **data)
arrays = np.load("test.npz")
x1 = arrays["x1"]
x2 = arrays["x2"]
print(x1)
print(x2)
