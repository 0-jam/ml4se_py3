# 最尤推定による回帰分析

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from dataset import create_dataset

## Parameters
# サンプルを取得する位置 x の個数
N = 100
# 多項式の次数
M = [0, 1, 3, 9]

### データ取得
## 最大対数尤度（Maximum log likelihood）を計算
def log_likelihood(dataset, f):
    dev = 0.0
    n = float(len(dataset))
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        dev += (y - f(x)) ** 2
    err = dev * 0.5
    beta = n / dev
    # この式はどこに対応？
    lp = -beta * err + 0.5 * n * np.log(0.5 * beta / np.pi)
    return lp

## 最尤推定で解を求める（解法は最小二乗法と同じ）
def resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    for i in range(0, m+1):
        p = dataset.x ** i
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis=1)
    # numpy.dot(): ベクトル・行列の内積
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    # p96 式3.21
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0.0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    sigma2 = 0.0
    for index, line in dataset.iterrows():
        sigma2 += (f(line.x) - line.y) ** 2
    sigma2 /= len(dataset)

    # np.sqrt(sigma2): p96 式3.22
    return (f, ws, np.sqrt(sigma2))

def main():
    train_set = create_dataset(N)
    test_set = create_dataset(N)
    df_ws = DataFrame()

    # 多項式近似の曲線を求めて表示
    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws, sigma = resolve(train_set, m)
        df_ws = df_ws.append(Series(ws,name="M = %d" % m))

        subplot = fig.add_subplot(2, 2, c + 1)
        subplot.set_xlim(-0.05, 1.05)
        subplot.set_ylim(-1.5, 1.5)
        subplot.set_title("M = %d" % m)

        # トレーニングセットを表示
        subplot.scatter(
            train_set.x,
            train_set.y,
            marker='o',
            color='blue',
            label=None
        )

        # 真の曲線を表示
        linex = np.linspace(0,1,101)
        liney = np.sin(2*np.pi*linex)
        subplot.plot(linex, liney, color='green', linestyle='--')

        # 多項式近似の曲線を表示
        linex = np.linspace(0,1,101)
        liney = f(linex)
        label = "Sigma = %.2f" % sigma
        subplot.plot(linex, liney, color='red', label=label)
        subplot.plot(linex, liney+sigma, color='red', linestyle='--')
        subplot.plot(linex, liney-sigma, color='red', linestyle='--')
        subplot.legend(loc=1)
    # p97 図3.5
    # fig.savefig("out/031-p97_fig3.5.png")

    # 多項式近似に対する最大対数尤度を計算
    df = DataFrame()
    train_mlh = []
    test_mlh = []
    for m in range(0,9): # 多項式の次数
        f, ws, sigma = resolve(train_set, m)
        train_mlh.append(log_likelihood(train_set, f))
        test_mlh.append(log_likelihood(test_set, f))
    df = pd.concat(
        [
            df,
            DataFrame(train_mlh, columns=['Training set']),
            DataFrame(test_mlh, columns=['Test set'])
        ],
        axis=1
    )
    df.plot(title='Log likelihood for N = %d' % N, grid=True, style=['-', '--'])
    # p98 図3.6
    # plt.savefig("out/031-p98_fig3.6.png")
    plt.show()

if __name__ == '__main__':
    main()
