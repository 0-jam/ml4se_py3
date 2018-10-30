# k平均法による画像の減色処理

import numpy as np
from numpy.random import randint
from PIL import Image
import sys

## Parameters
# 減色後の色数（任意の個数の色数を指定できます）
Colors = [2, 3, 5, 16]


## k平均法による減色処理
def run_kmeans(pixels, k):
    cls = [0] * len(pixels)

    # 代表色の初期値をランダムに設定
    center = []
    for i in range(k):
        center.append(np.array([randint(256), randint(256), randint(256)]))
    print("Initial centers:"),
    print(map(lambda x: x.tolist(), center))
    print("========================")
    distortion = 0.0

    # 最大50回のIterationを実施
    for iter_num in range(50):
        center_new = []
        for i in range(k):
            center_new.append(np.array([0, 0, 0]))
        num_points = [0] * k
        distortion_new = 0.0

        # E Phase: 各データが属するグループ（代表色）を計算
        for pix, point in enumerate(pixels):
            min_dist = 256 * 256 * 3
            point = np.array(point)
            for i in range(k):
                d = sum([x * x for x in point - center[i]])
                if d < min_dist:
                    min_dist = d
                    cls[pix] = i
            center_new[cls[pix]] += point
            num_points[cls[pix]] += 1
            distortion_new += min_dist

        # M Phase: 新しい代表色を計算
        for i in range(k):
            center_new[i] = center_new[i] / num_points[i]
        center = center_new
        print(map(lambda x: x.tolist(), center))
        print("Distortion: J = %d" % distortion_new)

        # Distortion（J：二重歪み）の変化が0.1%未満になったら終了
        if iter_num > 0 and distortion - distortion_new < distortion * 0.001:
            break
        distortion = distortion_new

    # 画像データの各ピクセルを代表色で置き換え
    # ここでastype(int)することで、RGB値を整数にしている
    for pix, point in enumerate(pixels):
        pixels[pix] = tuple(center[int(cls[pix])].astype(int))

    return pixels

def main():
    for k in Colors:
        fname = "data/photo.jpg" if (len(sys.argv) == 1) else (sys.argv[1])
        im = Image.open(fname)
        print("")
        print("========================")
        print("Number of clusters: K = %d" % k)
        # 画像ファイルの読み込み
        pixels = list(im.convert('RGB').getdata())
        # k平均法による減色処理
        result = run_kmeans(pixels, k)
        # 画像データの更新とファイル出力
        im.putdata(result)
        im.save("out/6/output%02d.png" % k, "PNG")

if __name__ == '__main__':
    main()
