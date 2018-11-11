# restoration_image.py
- 作成者：Funamoto
- リリース日：2018.6.22
- 更新日：2018.6.22

# restoration_image.pyの概要
1. imageのディレクトリにある画像をランダムに回転、平行移動、拡大、縮小させ、gene_imageに保存
2. １の画像を復元してまっすぐに戻す


# 動作環境
- Python 3.6.5
- OpenCV 3.4.0
- numpy 1.14.2


# ディレクトリ構成内容の説明
- `README.md`：このファイルです
- `restoration_image.py`：実行ファイル。
- `restoration_image.ipynb`：restoration_image.pyと同じコード。実行結果がジュピターの方が見やすいので念のため掲載
- `image`：回転平行移動をする前の元の画像が入っている。restoration_image.pyを実行すると、ここから画像をロードして処理を行う
- `gene_image`：restoration_image.pyで回転平行移動した後の画像が保存されるディレクトリ
- `restoration_image`：restoration_image.pyで復元した画像が保存されるディレクトリ


# 環境の構築

次のコマンドによって必要環境を構築できる。
```
# まず作業フォルダに移動します
cd restoration

# OpenCVのインストール
pip install opencv-contrib-python

# numpyのインストール
pip install numpy

```


# 実行

以下で実行する。
```
# 作業ディレクトリへ移動する
$ cd restoration

# 実行する
$ python restoration_image.py

# 実行後の画像は次のディレクトリに生成される
$ cd gene_image
$ cd restoration_image
```



# 使い方

## まずパラメータをセッティングしてください

`生成する画像の数`
num_image = ...


# 以下はランダムに画像を生成するときのパラメーターセッティングです。
# 復元をする時に設定するパラメーターはありません。
`生成する画像の一辺のサイズ（正方形を想定）`
image_size= ...

`以下の範囲内でランダムに画像を拡大、縮小。
　画像を縮小するときの最小%`
size_min = ...

`画像を拡大するときの最大%`
size_max = ...

`画像を回転させる時の最大角度`
ang_max = ...



# 更新履歴
2018.6.22 - リリース
