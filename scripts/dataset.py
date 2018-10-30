import numpy as np
from pandas import Series, DataFrame
from numpy.random import normal

## データセット {x_n, y_n} (n = 1...N) を用意
# ここでは標準偏差0.3のサイン波(sin(2πx))
def create_dataset(num):
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(num):
        x = float(i) / float(num - 1)
        # numpy.normal : 正規分布を求める
        y = np.sin(2 * np.pi * x) + normal(scale=0.3)
        dataset = dataset.append(Series([x, y], index=['x', 'y']), ignore_index=True)
    return dataset
