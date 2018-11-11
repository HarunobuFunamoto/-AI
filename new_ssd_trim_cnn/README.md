# corner
- 作成者：Funamoto
- リリース日：2018.7.18
- 更新日：2018.7.18


# 動作環境
- Python 3.6.5
- OpenCV 3.4.0
- numpy 1.14.2
- keras 2.0.5


# ディレクトリ構成内容の説明
- `README.md`：このファイルです
- `training_SSD`：SSDで学習するためのファイル
- `XML_pkl.py`:training_SSDで必要なpklファイルに変換するコード。xmlファイルをpklファイルに変換。
- `load_ssd_predict.py`：ここの関数で重みをロードしてSSDで画像を予測する。triming_ssd_image.pyの中で実行される
- `triming_ssd_image.py`：load_ssd_predict.pyを実行するファイル。その関数から帰ってきた画像と座標を使って、画像を切り抜く。npyファイルで保存する。
- `create_allmoji.py`：漢字、ひらがな、カタカナ、数字が格納されたリストを返す関数がある。cre_allmoji_image.pyで実行される
- `cre_allmoji_image.py`：create_allmoji.pyを実行し、全ての文字を印字した画像を生成する。npyファイルで出力
- `cnn_allmoji.py`：cre_allmoji_image.pyで作ったnpyファイルをロードして学習。モデルと重みを保存。
- `load_cnn_allmoji.py`：最終工程。cnn_allmoji.pyで保存した重みとモデルをロードし、triming_ssd_image.pyで切り取った画像にどんな文字が書いているか出力する。

そのほかのファイルSSD作動時に必要なもので、特に使わない。



# SSD学習編
1. labelImgというフリーソフトを用いてバウンディングボックスを作成。xmlというディレクトリにxmlファイルの保存先を指定して、手でボックスを作る
2. XML_pkl.pyというファイルでxmlからpklファイルに変換
3. training_SSD.ipynbで二箇所だけ変更して学習。一箇所めはpklファイル指定。二箇所めは真ん中あたりにある画像のファイルパス指定。あとは実行すれば学習できる。


# CNN学習編
1. cre_allmoji_image.pyを実行。漢字、ひらがな、カタカナ、数字の文字を印字した画像を生成し、all_mojiディレクトリへnpyファイルで保存。

以下で実行
$ python ceate_allmoji_image.py

2. cnn_allmoji.pyを実行すると、画像になんの文字が書かれているか出力される。cre_allmoji_image.pyで作ったnpyファイルを読み込み学習。重みとモデルをcnn_model_weightへ保存。

以下で実行
$ python cnn_allmoji.py



# 重みをロードしてSSD→→画像切り取り→→重みとモデルをロードしてCNNで学習
## 1 重みをロードしてSSDで予測
```
load_ssd_predict.pyのSSD_predict()という関数の中で、先ほど学習した重みを使ってSSDでバウンディングボックスを予測し、座標の抽出が行われる
→このファイルで読み込む重みと予測したい画像を指定しなければならない。
SSDで学習した重みはcheckpointsに入っている。
画像はパスを変えて指定するが、今回は三枚まで同時に予測できるようにしている。
もし予測したい画像を増やしたい時は以下のコードをコピペして画像パスだけ変更する。

img_path = 'predict_image/000001.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())

なお、SSD_predict関数は次の行程のファイルのtriming_ssd_image.pyの中で実行される。

この関数の入力は、重みと予測したい画像。
出力は、予測したい画像と座標。次の行程で画像の切抜きとして使われる。


```
## 2 SSDで抜き出した座標に従い、画像を切り抜き、切り抜いた画像をnpyファイルで保存

```
１のSSDで予測するSSD_predict関数はこのスクリプト内で実行される為、

以下で実行
$ python triming_ssd_image.py

とコマンドを打てば、SSDによる予測〜画像を切り出し〜npy保存まで一気にやってくれる。
実行した後に色々ブワーっと出るが、最後に OK　と表示されれば上手く実行されて、作業が完了している。

入力：SSD_predict関数で出力した、予測したい画像と座標
出力：切り取られた画像が、ssd_trim_imgディレクトリにnpyファイルが保存されている

```


## 3 重みとモデルをロードしてCNNで学習

```
最終行程。以下で実行
$ python load_cnn_allmoji.py

このファイルでは、
1. 文字を学習したCNNの重みとモデルを指定する。（CNN学習編の重みとモデルを指定、cnn_model_weightに格納されている）
2. ２で画像を切り出したnpyファイルを指定する→この画像をpredictしてどんな画像が書かれているか出力する。 ssd_trim_imgディレクトリにnpyファイルが保存されている

入力：CNNで学習した重み、モデル、２で切り出した画像
出力：２で切り出した画像に何の文字が書かれているかプリント

```


# パラメーターの設定

## cnn_allmoji_image.pu

```
#  漢字のリストが格納されている
text = cre_moji()
# 作成する画像のサイズ指定（正方形）
image_size = 200
# 回転する角度の範囲を指定
ang_min= -100
ang_max = 100
# 以下の範囲内でランダムに画像を拡大、縮小。
#画像の縮小率
size_min = 60
# 画像の拡大率はこのままで
size_max = 100
# 一文字につき、何枚回転、平行移動、縮小した画像を作成するか指定。回転も移動も縮小もしない画像は1文字につき1枚は作成していて、
#それ以外に何枚画像を変形させるか指定する
num_image = 1
tate =50
yoko =50 
```

## cnn_allmoji.py
```
batch_size = 1
epoch = 1

```

## load_cnn_allmoji.py
```
# 使用する重みのファイル名
#filename = sys.argv[1]
# 使用するモデルのファイル名
modelname = 'cnn_model_weight/model.json'
# 使用する重みのファイル名
weight = 'cnn_model_weight/CNN_.00-0.00.hdf5'
# imagesに予測したいnpyファイルを与える=切り取った免許証の画像　
images = 'ssd_trim_img/trim_menkyo.npy'

```

## triming_ssd_image.py
```
image_size = 200
```

## training_SSD.ipync
```
pklファイルを指定
gt = pickle.load(open('/Users/user/SSD/12moji_menkyo.pkl', 'rb'))

画像ディレクトリを指定
path_prefix = '/Users/user/Desktop/12moji/'

```


## XML_pkl.py
```
# labelImgで作ったxmlがある保存先のディレクトリを指定
data = XML_preprocessor('xml/').data

# SSD学習するときにはこのtrain.pklファイルを入れる
pickle.dump(data,open('pkl/train.pkl','wb'))

```


# 更新履歴
2018.7.18 - リリース
