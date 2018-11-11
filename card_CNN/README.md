# 帳票分類
- 作成者：Funamoto
- リリース日：2018.6.21
- 更新日：2018.6.21


# card_classify.py の概要
1. CNNで学習・予測させたい画像を images のディレクトリ内に保存。imagesの中にパスポート、マイナンバー、住民票、保険証、免許証のディレクトリがあるので、画像をそれぞれ分けて保存する。
2. generate_train_test.pyのファイルにあるgene_train_test関数でimages内にある画像を読み込む。指定のピクセルにリサイズし、学習用とテスト用に分けてgenerate_fileに保存
3. 1のファイルをgenerate_fileからロードし、CNNで学習と予測を行う。
4. 2の結果をaccuracyと適合率、再現率、F１値、混同行列を表示。加えて、予測した帳票と正解ラベルの帳票も並べて表示する。


# 動作環境
- Python 3.6.5
- numpy 1.14.2
- pandas 0.22.0
- Pillow 5.0.0
- Keras 2.0.5
- scikit-learn 0.19.1
- glob2 0.6


# ディレクトリ構成内容の説明
- `README.md`：このファイル
- `cards_classify.py`：実行ファイル。データ用と学習用に画像を分けて保存し、それらをCNNで画像分類を行う
- `generate_train_test.py`：imagesから画像を読み込みリサイズし、データ用と学習用に分けてgenerate_fileに保存する。cards_classify.py内で関数としてこのファイルを呼び出され実行されている。
- `generate_file`：generate_train_test.pyによって分割された画像が保存されるディレクトリ
- `images`：このディレクトリの中には、CNNの学習・予測に使うパスポート、マイナンバー、住民票、保険証、免許証の画像が格納されている。generate_train_test.pyではこのディレクトリから画像を読み込み、学習用とテスト用に分類している。



# 環境の構築

次のコマンドによって必要環境を構築。
```
# まず作業フォルダに移動
cd card_CNN

# numpyのインストール
pip install numpy

# pillowのインストール
pip install pillow

# pandasのインストール
pip install pandas

# glob2のインストール
pip install glob2

# scikit-learnのインストール
pip install scikit-learn

＃kerasのインストール
pip install keras==2.0.5

# 準備完了ー
```


# 実行

`作業ディレクトリへ移動する`
$ cd card_CNN

`実行。実行するたびに画像を学習用とテスト用に分類して保存し、それらをロードしてCNNで学習予測する。`
$ python cards_classify.py

`実行結果`
生成された画像は次のディレクトリに保存される
$ cd generate_file


`もし、generate_fileのディレクトリに保存したデータを使って、CNNで学習予測だけしたい場合`
cards_classify.pyの26行目をコメントアウトすると、学習予測だけ行える。
実行例：　# gene_train_test(image_size)



# 使い方

```
cards_classify.pyでパラメータをセッティング　
```

1. エポック数
14行目：epoch =　
ここでCNNで学習するときのエポック数を指定する。

2. バッチサイズ
15行目：batch_size =　
NNで学習するときのバッチサイズを指定する。


3. 画像の縦と横のピクセル数
16行目：image_size =
CNNで学習させる画像は縦横同じサイズでないといけないので、ここで揃える縦横のピクセル数を指定する。


4. 保存するファイル名
18行目：file =
generate_fileに保存するファイル名を指定。同じ名前だと上書き保存されてしまう為、違う名前にすると上書き保存を免れる。



# 更新履歴
2018.6.21 - リリース
