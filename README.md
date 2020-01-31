# Machine Learning for Software Engineers (Python 3 compatible)

Sample scripts of "Machine Learning for Software Engineers" ([Original scripts][orig] were written in Python 2).

All scripts has been ported to Python 3.

---

1. [Environment](#environment)
   1. [Software](#software)
1. [Installation](#installation)
1. [Changes](#changes)

---

## Environment

### Software

- Python 3.6.7 on Windows 10 1803

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

[orig]: https://github.com/enakai00/ml4se
[py3div]: https://docs.python.org/3/howto/pyporting.html#division
[astype]: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.astype.html
[pylzma]: https://docs.python.org/3/library/lzma.html
[pep822]: https://www.python.org/dev/peps/pep-0008/#source-file-encoding
