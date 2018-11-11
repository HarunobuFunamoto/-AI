# lstm_classify.py
- 作成者：Funamoto
- リリース日：2018.7.31
- 更新日：2018.7.31

# 概要
1. 免許証に書かれている文字とラベル名（氏名、住所など）を記したcsvファイルを読み込む
2. 重みをロードしてSSDで座標を保存したnpyファイル（load_ssd_predict2.pyで保存したもの）をロードする
3. １の文字と２の座標を使って、LSTMに学習。
4. 免許証に書かれている文字列を予測。



# 動作環境
- Python 3.6.5
- numpy 1.14.2
- keras 2.0.5


# ディレクトリ構成内容の説明
- `README.md`：このファイルです
- `lstm_classify.py`：概要を全て実行するファイル。このフォルダ内から扱うデータは名前と生年月日、交付年月日、有効期限、番号だけだが、csvと座標のnpyファイルさえ変えれば、免許証全体まで範囲を広げられる。
- `menkyo_info.csv`：CNNから文字を出力した文字列とラベル名（氏名、住所など）が格納されているcsvファイル。僕の免許証の中の、名前と生年月日、交付年月日、有効期限、番号の部分の文字列とラベル名が格納されている。
- `zahyou.npy`：名前と生年月日、交付年月日、有効期限、番号の部分の座標が格納されている。本来はload_ssd_predict2.pyで保存したtrim/image_point.npyファイルをロードすべきだが、今回は名前と生年月日、交付年月日、有効期限、番号だけの入力に限定しているので、それに対応する座標が格納されたzahyou.npyを作成した。このファイルを概要２でロードする。



# 環境の構築

次のコマンドによって必要環境を構築できる。
```
まず作業フォルダに移動します
$ cd lstm_fts

numpyのインストール
$ pip install numpy

kerasのインストール
$ pip install keras==2.0.5
```


# 実行


```
作業ディレクトリへ移動する
$ cd lstm_fts

概要を全て実行。
$ python lstm_classify.py
```


# 教師データの追加方法
- `menkyo_info.csv`のようなcsvファイルを作成する。今回は免許証の画像を見ながら書かれている文字を一列目に、ラベルを二列目に書いてcsvを作成した。
- 座標の情報が格納されている`zahyou.npy`は、LSTMが動くかどうか確認するために、座標に見立てて適当に作ったファイル。本来はSSDで予測して作られたtrim/image_point.npyをロードすればいいので、別途用意する必要はない。
　 



# パラメーターの設定

## CNNのパラメータセッティング（cnn_corner.py内で設定）

```
バッチサイズ
batch_size = ...

エポック数
epoch = ...
```

## lstm_classify.pyの引数セッティング（cnn_corner.py内で設定）
```
epoch = 30

batch_size ,neurons,neurons2,neurons3,embedding_vecor_length,learning_rateはエポック３０での試行錯誤の末、一番良かった組み合わせにしています

batch_size = 2

出力する時の次元数。
LSTM１層目
neurons = 70
LSTM２層目
neurons2 = 70
LSTM３層目
neurons3 = 100

出力する密ベクトルの次元数
embedding_vecor_length = 2

アダムの学習率
learning_rate = 0.5

```



# 更新履歴
2018.7.31 - リリース
