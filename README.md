# Machine Learning for Software Engineers (Python 3 compatible)

Sample scripts of "Machine Learning for Software Engineers" ([Original scripts][orig] were written in Python 2).

All scripts has been ported to Python 3.

---

1. [Environment](#environment)
    1. [Software](#software)
1. [Todo](#todo)
1. [Installation](#installation)
1. [Changes](#changes)

---

## Environment

### Software

- Python 3.6.7 on Windows 10 1803

## Todo

- [ ] 実行時にいくつか警告が出ているので原因を調べる
- [ ] 補足
    - 本に書かれている中で、プログラム中で使われてる数式は対応させたい
- [x] Python3互換
    - [x] `061-k_means.py`で型変換エラー
        - > TypeError: integer argument expected, got float
        - 代表色計算時にRGB値に浮動小数点が入っていた
            - `numpy.array`の全要素を[astype(int)][astype]で整数に変換
                - 計算終了後に変換するので、誤差は最小限…のはず
        - Python3で整数同士の除算をすると、結果が浮動小数点になるようになった（[参考][py3div]）

## Installation

```bash
# pyenvを使ってPythonをインストールする場合、Pythonのビルド前にtk-devをインストールする必要がある（python-tkは不要）
# 入れていないと"ModuleNotFoundError: No module named 'tkinter'"
$ sudo apt install tk-dev python-tk
$ pip install matplotlib numpy scipy pandas pillow
```

## Changes

- ファイル先頭の`# -*- coding: utf-8 -*-`を削除
    - Python3ではデフォルトがUTF-8で、デフォルトのエンコーディングを明示するのは[非推奨][pep822]のため
    - > Files using ASCII (in Python 2) or UTF-8 (in Python 3) should not have an encoding declaration.
- データセット生成処理をモジュール化
- `scripts/061-k_means.py`の対象ファイルを引数で指定できるようにした
- テストデータをXZ(LZMA2)で圧縮
    - そのままだとGitHubに上げられるサイズ(100MB)を超えていたため
        - [元のリポジトリ][orig]ではGZipで圧縮されている
    - [LZMA][pylzma]モジュールにより、圧縮ファイルを展開しなくてもプログラムを実行できるようにした

---

```bash
## 11÷3＝3（整数）になる言語
# 2つの数字の少なくとも片方に小数点があれば、計算結果は浮動小数点になる
# Python2
$ python2 -c "print(11/3)"
3
# Ruby
$ ruby -e "puts 11/3"
3
# Bash（$ expr 11 / 3でも同じ）
$ echo $(( 11 / 3 ))
3
# 浮動小数点を含む計算はできない
$ echo $(( 11.0 / 3.0 ))
-bash: 11.0 / 3.0 : syntax error: invalid arithmetic operator (error token is ".0 / 3.0 ")
$ expr 11.0 / 3.0
expr: non-integer argument

## 11÷3＝3.6666...になる
# Python3
$ python3 -c "print(11/3)"
3.6666666666666665
# Perl5
$ perl -e "print(11/3)"
3.66666666666667
# Perl6
> perl6 -e "say(11/3)"
3.666667
# PHP
$ php -r "echo 11/3;"
3.6666666666667
```

[orig]: https://github.com/enakai00/ml4se
[py3div]: https://docs.python.org/3/howto/pyporting.html#division
[astype]: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.astype.html
[pylzma]: https://docs.python.org/3/library/lzma.html
[pep822]: https://www.python.org/dev/peps/pep-0008/#source-file-encoding
