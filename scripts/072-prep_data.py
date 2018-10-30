# 手書き文字サンプルの抽出

import lzma
import re
from subprocess import Popen, PIPE

def main():
    ## Parameters
    # 抽出する文字数
    Num = 600
    # 抽出する数字（任意の個数の数字を指定可能）
    Chars = '[036]'

    with lzma.open('data/train-labels.txt.xz') as labels, lzma.open('data/train-images.txt.xz') as images, open('data/sample-labels.txt', 'w') as labels_out, open('data/sample-images.txt', 'w') as images_out:
        chars = re.compile(Chars)

        while True:
            label = labels.readline().decode()
            image = images.readline().decode()
            if (not image) or (not label):
                break
            if not chars.search(label):
                continue

            line = ''
            for c in image.split(" "):
                if int(c) > 127:
                    line += '1,'
                else:
                    line += '0,'
            line = line[:-1]
            labels_out.write(label)
            images_out.write(line + '\n')
            Num -= 1
            if Num == 0:
                break

    with open('data/sample-images.txt', 'r') as images, open('out/samples.txt', 'w') as samples:
        c = 0

        while True:
            line = images.readline()
            if not line:
                break
            x = 0
            for s in line.split(','):
                if int(s) == 1:
                    samples.write('#')
                else:
                    samples.write(' ')
                x += 1
                if x % 28 == 0:
                    samples.write('\n')
            c += 1
            if c == 10:
                break

if __name__ == '__main__':
    main()
