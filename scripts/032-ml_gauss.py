# 最尤推定による正規分布の推定

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import normal
from scipy.stats import norm

def main():
    fig = plt.figure()
    # サンプル数：2, 4, 10, 100個
    # 平均0, 標準偏差1
    for c, datapoints in enumerate([2, 4, 10, 100]):
        ds = normal(loc=0, scale=1, size=datapoints)
        # 平均の推定値
        # p104 式3.34
        mu = np.mean(ds)
        # 標準偏差の推定値
        # p104 式3.35
        sigma = np.sqrt(np.var(ds))

        subplot = fig.add_subplot(2, 2, c+1)
        subplot.set_title("N = %d" % datapoints)
        # 真の曲線を表示
        linex = np.arange(-10, 10.1, 0.1)
        orig = norm(loc=0, scale=1)
        subplot.plot(linex, orig.pdf(linex), color='green', linestyle='--')
        # 推定した曲線を表示
        est = norm(loc=mu, scale=sigma)
        label = "Sigma = %.2f" % sigma
        subplot.plot(linex, est.pdf(linex), color='red', label=label)
        subplot.legend(loc=1)
        # サンプルの表示
        subplot.scatter(ds, orig.pdf(ds), marker='o', color='blue')
        subplot.set_xlim(-4, 4)
        subplot.set_ylim(0)
    # p105 図3.11
    # 緑の破線が正解で、赤の実線が推定されたグラフ
    # ガウシアン関数
    # fig.savefig("out/032-p105_fig3.11.png")
    plt.show()

if __name__ == '__main__':
    main()
