# main code만 실행시 ModuleNotFoundError: No module named 'dezero'
# -> 해결방법은 현재 실행중인 파일 디렉토리의 부모 디렉토리(..)를 모듈 검색 경로에 추가한다.
# globals(): dict of global variables and symbols of current program
if "__file__" in globals():
    import os, sys

    # os.path.dirname(__file__) : return 현재 실행중인 파이썬 파일의 path
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# main code
from dezero import Variable


# test functions for optimization
def sphere(*x):
    from functools import reduce

    y = reduce(lambda i, j: i**2 + j**2, x)
    return y


def matyas(x, y):
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
    return y
