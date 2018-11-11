
from keras.models import Sequential,Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,os
from FTS_images import generate_images


# パラメーター一覧
baseSaveDir = "./cnn/"


# 画像の生成は他のファイルからインポートして行う。画像生成の関数は2種類用意しました。


#  画像を作る関数 　def generate_images(num_image, image_size, size_min, size_max, ang_max):

    # この関数はcross_validation.train_test_splitで自動的に学習用とテスト用に振り分ける関数
    #  実行後、作った画像の数と学習用の画像の数とテスト用の画像の数をプリントする

    #  生成する画像の数：num_image

    #  生成する画像の一辺のサイズ（正方形を想定）：image_size　

    # 以下の範囲内でランダムに画像を拡大、縮小。
    #  画像を縮小するときの最小ピクセル:size_min
    #  画像を拡大するときの最大ピクセル:size_max

    # 画像を回転させる時の最大角度:ang_max

    #  平行移動に関してはmove = int( (bg_image_size - p_image_size) / 2)　という式で移動する条件を定めている


#バッチサイズ
batch_size = 32

# エポック数
epoch = 10


# どちらか一方をコメントアウトして画像生成してください。どちらを使うかはお任せします。
generate_images(10, 200, 50, 200, 359)




# 学習データとテストデータを分ける --- (※3)
data_train, data_test, label_train_list, label_test_list = np.load('data/train_test.npy')

#print(label_train_list)

#  ラベルには[サイズ、移動させた縦方向、移動させた横方向、回転させた角度] の順で格納されている。
#  以下はそれぞれを取り出して、リストに格納する作業

label_train_size = []
label_train_row = []
label_train_col = []
label_train_angle = []
for train in label_train_list:
    label_train = train
    #print('label_train',label_train)

    label_tr_size = label_train[0]
    label_tr_row = label_train[1]
    label_tr_col = label_train[2]
    label_tr_angle = label_train[3]

    label_train_size.append( label_tr_size)
    label_train_row.append( label_tr_row)
    label_train_col.append( label_tr_col)
    label_train_angle.append( label_tr_angle)


label_test_size = []
label_test_row = []
label_test_col = []
label_test_angle = []
for test in label_test_list:
    label_test = test
    #print('label_test',label_test)

    label_te_size = label_test[0]
    label_te_row = label_test[1]
    label_te_col = label_test[2]
    label_te_angle = label_test[3]

    label_test_size.append(label_te_size )
    label_test_row.append(label_te_row )
    label_test_col.append(label_te_col )
    label_test_angle.append(label_te_angle )



data_train = np.array(data_train)
data_test  = np.array(data_test)

label_test_size = np.array(label_test_size)
label_test_row = np.array(label_test_row)
label_test_col = np.array(label_test_col)
label_test_angle = np.array(label_test_angle)

label_train_size = np.array(label_train_size)
label_train_row = np.array(label_train_row)
label_train_col = np.array(label_train_col)
label_train_angle = np.array(label_train_angle)



#  正規化
data_train = data_train.astype("float") / 256
data_test  = data_test.astype("float") / 256




def build_model(in_shape):
    i = Input(shape=(in_shape))
    x = Convolution2D(32, 3, 3,border_mode='same',input_shape=in_shape)(i)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.7)(x)

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Convolution2D(32, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)


    x = Convolution2D(64, 3, 3)(x)
    x =Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.7)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(100)(x)
    x = Activation('relu')(x)

    x = Dense(20)(x)
    x = Activation('relu')(x)

    output1 = Dense(1,activation='linear',name='output1')(x)
    output2 = Dense(1,activation='linear',name='output2')(x)
    output3 = Dense(1,activation='linear',name='output3')(x)
    output4 = Dense(1,activation='linear',name='output4')(x)


    # 上の流れをインスタンス化→コンパイルするために
    multiModel = Model(inputs= i , outputs = [output1, output2, output3, output4])

    multiModel.compile(loss={'output1': 'mean_squared_error',
                         'output2': 'mean_squared_error',
                         'output3': 'mean_squared_error',
                         'output4': 'mean_squared_error'},
                   optimizer='Adam')


    return multiModel


model = build_model(data_train.shape[1:])

chkpt = os.path.join(baseSaveDir, 'CNN_.{epoch:02d}-{loss:.2f}.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=True, mode='auto')

#label_train_size
model.fit(data_train,  {'output1': label_train_size,
                          'output2': label_train_row,
                          'output3': label_train_col,
                          'output4': label_train_angle}, batch_size = batch_size, nb_epoch = epoch ,callbacks=[cp_cb])




model_json_str = model.to_json()
open('cnn/model.json', 'w').write(model_json_str)




output1, output2,output3,output4= model.predict(data_test)


print('\nlabel_size', 'predicted', sep='\t')
for idx, pre_label_size in enumerate(output1):
    for pre_size in pre_label_size:
            print(label_test_size[idx], ' ',pre_size, sep='\t')

print('\n')


 #  label_train_row
print('\nlabel_row', 'predicted', sep='\t')
for idx, pre_label_row in enumerate(output2):
    for pre_row in pre_label_row:
            print(label_test_row[idx], ' ',pre_row, sep='\t')


print('\n')


#  label_train_col
print('\nlabel_col', 'predicted', sep='\t')
for idx, pre_label_col in enumerate(output3):
    for pre_col in pre_label_col:
            print(label_test_col[idx], ' ',pre_col, sep='\t')


print('\n')


#label_train_angle
print('\nlabel_angle', 'predicted', sep='\t')
for idx, pre_label_angle in enumerate(output4):
    for pre_angle in pre_label_angle:
            print(label_test_angle[idx], ' ',pre_angle, sep='\t')

