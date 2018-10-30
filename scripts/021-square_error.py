# 誤差関数（最小二乗法）による回帰分析

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from dataset import create_dataset

## Parameters
# サンプルを取得する位置 x の個数
N = 100
# 多項式の次数
# 0次式はただの定数、1次式は傾きのある直線
# 9次式はこの場合すべての点を通る
# パラメータは(M[i] + 1)個存在する
#   f(x) = 3x + 2
M = [0, 1, 3, 9]

### データ取得
## 平方根平均二乗誤差（Root mean square error）を計算
# Pythonは関数を後ろのほうに定義できない
def rms_error(dataset, f):
    err = 0.0
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        # p67 式(2.17)
        err += 0.5 * (y - f(x)) ** 2

    # p67 式(2.20)
    # 式(2.17)に式(2.16)を代入してp70 式(2.21)を出している
    # E_RMS（戻り値）が、グラフのとっている値から平均でどのくらい離れているかを示している
    return np.sqrt(2 * err / len(dataset))

## 最小二乗法(least squares method)で解を求める
def lsm_resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    # p67 式(2.19)
    for i in range(0, m + 1):
        p = dataset.x ** i
        p.name = "x ** %d" % i
        phi = pd.concat([phi, p], axis=1)

    # p67 式(2.18)
    # この式によって多項式を決定することができる
    # T: 転置行列
    # 元の行列は多項式の係数
    # 自身の転置行列と自身との積をとって、それの逆行列を求めている
    # …で、その逆行列と転置行列に、目的変数tをかけている
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    # 返り値に関数を持たせる
    # Pythonでは関数がオブジェクト扱いなので、戻り値として関数そのものを返すことができる
    return (f, ws)

## Main
def main():
    train_set = create_dataset(N)
    test_set = create_dataset(N)
    df_ws = DataFrame()

    # 多項式近似の曲線を求めて表示
    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws = lsm_resolve(train_set, m)
        df_ws = df_ws.append(Series(ws, name="M = %d" % m))

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
        linex = np.linspace(0, 1, 101)
        liney = np.sin(2 * np.pi * linex)
        subplot.plot(linex, liney, color='green', linestyle='--')

        # 多項式近似の曲線を表示
        linex = np.linspace(0,1,101)
        liney = f(linex)
        label = "E(RMS)=%.2f" % rms_error(train_set, f)
        subplot.plot(linex, liney, color='red', label=label)
        subplot.legend(loc=1)

    # 係数の値を表示
    # p69 図2.3
    # N = 100の場合：p80 図2.11
    print("Table of the coefficients")
    print(df_ws.transpose())
    # N = 100の場合：p80 図2.12（2.2とあまり変わらない）
    # fig.savefig("out/021-p68_fig2.2.png")

    # トレーニングセットとテストセットでの誤差の変化を表示
    df = DataFrame(columns=['Training set','Test set'])
    # 多項式の次数の分だけ繰り返す
    for m in range(0, 10):
        f, ws = lsm_resolve(train_set, m)
        train_error = rms_error(train_set, f)
        test_error = rms_error(test_set, f)
        df = df.append(
                Series([train_error, test_error], index=['Training set', 'Test set']),
                ignore_index=True
            )

    # p77 図2.8
    # 次数Mが3以上になったあたりで、テストセットに対する誤差の減りが鈍くなる（0.3くらいに収束する）
    # トレーニングセットに対してはN = 10の場合M = 9で完全に正確（トレーニングセットの学習結果をトレーニングセットとの誤差で比較してるから当然だが）
    # →過学習（オーバーフィッティング）：トレーニングセットに特化した汎用性のない結果を得てしまう
    df.plot(title='RMS Error', style=['-', '--'], grid=True, ylim=(0, 0.9))
    # plt.savefig("out/021-p77_fig2.8.png")
    plt.show()

if __name__ == '__main__':
    main()
