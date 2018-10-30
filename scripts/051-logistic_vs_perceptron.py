# ロジスティック回帰とパーセプトロンの比較

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import rand, multivariate_normal
pd.options.mode.chained_assignment = None

## Parameters
# 両クラス共通の分散（4種類の分散で計算を実施）
Variances = [5, 10, 30, 50]

# データセット {x_n, y_n, type_n} を用意
def prepare_dataset(variance):
    n1 = 10
    n2 = 10
    mu1 = [7, 7]
    mu2 = [-3, -3]
    cov1 = np.array([
        [variance, 0],
        [0, variance]
    ])
    cov2 = np.array([
        [variance, 0],
        [0, variance]
    ])

    df1 = DataFrame(multivariate_normal(mu1, cov1, n1),columns=['x', 'y'])
    df1['type'] = 1
    df2 = DataFrame(multivariate_normal(mu2, cov2, n2), columns=['x', 'y'])
    df2['type'] = 0
    df = pd.concat([df1, df2],ignore_index=True)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    return df

# ロジスティック回帰
def run_logistic(train_set, subplot):
    # p144 式5.11?
    w = np.array([
        [0],
        [0.1],
        [0.1]
    ])
    phi = train_set[
        ['x', 'y']
    ]
    phi['bias'] = 1
    # p146 式5.16
    # TODO: FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
    #   そのまま変えても動かない
    phi = phi.as_matrix(columns=['bias', 'x', 'y'])
    # p146 式5.15
    t = train_set[
        ['type']
    ]
    t = t.as_matrix()

    # 最大30回のIterationを実施
    for i in range(30):
        # IRLS法(Iteratively Reweighted Least Squares：反復重み付け最小二乗法)によるパラメータの修正
        y = np.array([])
        for line in phi:
            a = np.dot(line, w)
            y = np.append(y, [1.0 / (1.0 + np.exp(-a))])
        # p146 式5.18
        # diag: 対角行列
        # TODO: RuntimeWarning: overflow encountered in exp
        #   原因究明
        r = np.diag(y * (1 - y))
        y = y[np.newaxis, :].T
        tmp1 = np.linalg.inv(np.dot(np.dot(phi.T, r), phi))
        tmp2 = np.dot(phi.T, (y - t))
        w_new = w - np.dot(tmp1, tmp2)
        # パラメータの変化が 0.1% 未満になったら終了
        # p147 式5.19
        if np.dot((w_new - w).T, (w_new - w)) < 0.001 * np.dot(w.T, w):
            w = w_new
            break
        w = w_new

    # 分類誤差の計算
    w0, w1, w2 = w[0], w[1], w[2]
    err = 0.0
    for index, point in train_set.iterrows():
        x, y, type = point.x, point.y, point.type
        type = type * 2 - 1
        if type * (w0 + (w1 * x) + (w2 * y)) < 0:
            err += 1
    err_rate = err * 100 / len(train_set)

    # 結果を表示
    xmin, xmax = (train_set.x.min() - 5), (train_set.x.max() + 10)
    linex = np.arange(xmin - 5, xmax + 5)
    liney = - linex * w1 / w2 - w0 / w2
    label = "ERR %.2f%%" % err_rate
    subplot.plot(linex, liney, label=label, color='blue')
    subplot.legend(loc=1)

# パーセプトロン
def run_perceptron(train_set, subplot):
    w0 = w1 = w2 = 0.0
    bias = 0.5 * (train_set.x.abs().mean() + train_set.y.abs().mean())

    # Iterationを30回実施
    for i in range(30):
        # 確率的勾配降下法によるパラメータの修正
        for index, point in train_set.iterrows():
            x, y, type = point.x, point.y, point.type
            type = (type * 2) - 1
            if type * ((w0 * bias) + (w1 * x) + (w2 * y)) <= 0:
                w0 += type * bias
                w1 += type * x
                w2 += type * y
    # 分類誤差の計算
    err = 0.0
    for index, point in train_set.iterrows():
        x, y, type = point.x, point.y, point.type
        type = (type * 2) - 1
        if type * ((w0 * bias) + (w1 * x) + (w2 * y)) <= 0:
            err += 1
    err_rate = err * 100 / len(train_set)

    # 結果を表示
    xmin, xmax = (train_set.x.min() - 5), (train_set.x.max() + 10)
    linex = np.arange(xmin - 5, xmax + 5)
    liney = - linex * w1 / w2 - bias * w0 / w2
    label = "ERR %.2f%%" % err_rate
    subplot.plot(linex, liney, label=label, color='red', linestyle='--')
    subplot.legend(loc=1)

# データを準備してロジスティック回帰とパーセプトロンを実行
def run_simulation(variance, subplot):
    train_set = prepare_dataset(variance)
    train_set1 = train_set[train_set['type'] == 1]
    train_set2 = train_set[train_set['type'] == 0]
    ymin, ymax = (train_set.y.min() - 5), (train_set.y.max() + 10)
    xmin, xmax = (train_set.x.min() - 5), (train_set.x.max() + 10)
    subplot.set_ylim([ymin - 1, ymax + 1])
    subplot.set_xlim([xmin - 1, xmax + 1])
    subplot.scatter(train_set1.x, train_set1.y, marker='o', label=None)
    subplot.scatter(train_set2.x, train_set2.y, marker='x', label=None)

    run_logistic(train_set, subplot)
    run_perceptron(train_set, subplot)

def main():
    fig = plt.figure()
    plt.suptitle('Blue: Logistic Regression, Red: Perceptron')
    for c, variance in enumerate(Variances):
        subplot = fig.add_subplot(2, 2, c + 1)
        run_simulation(variance, subplot)
    plt.show()
    # fig.savefig("out/051-p148_fig5.5.png")

if __name__ == '__main__':
    main()
