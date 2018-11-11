from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
from PIL import Image
import glob
import keras.backend as K
from sklearn import cross_validation , metrics
from sklearn.metrics import confusion_matrix,f1_score ,recall_score ,precision_score,classification_report
from generate_train_test import gene_train_test


epoch = 50
batch_size = 9
image_size = 50
# 保存したファイル名を記入
file = 'card.npy'


label_card = {0: '免許証', 1: '住民票', 2: 'パスポート', 3: '保険証', 4: 'マイナンバー'}
card = ['免許証','住民票','パスポート','保険証','マイナンバー']
nb_card = len(card)

#ここで学習用とテスト用のデータとラベルを生成する。データとラベルの生成はせずに学習だけしたい場合は以下をコメントアウトする。
gene_train_test(image_size)


# 学習データとテストデータを分けてロード
data_train, data_test, label_train, label_test = np.load('generate_file/'+file)

data_train2 = np.array(data_train)
data_test2  = np.array(data_test)
label_train2 = np.array(label_train)
label_test2 = np.array(label_test)


#  正規化
data_train = data_train2.astype("float") / 256
data_test  = data_test2.astype("float") / 256
label_train = np_utils.to_categorical(label_train2, nb_card)
label_test = np_utils.to_categorical(label_test2, nb_card)



def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nb_card))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'])

    return model


model = build_model(data_train.shape[1:])
model.fit(data_train, label_train, batch_size = batch_size, nb_epoch = epoch)
score = model.evaluate(data_test, label_test)
pred = model.predict(data_test)
predicted = np.argmax(pred, axis=1)



print('loss = ' , score[0])
print('accuracy =', score[1])
print('\nreport\n' , classification_report(np.argmax(label_test, axis=1), predicted))
print('\nconfusion matrix\n',confusion_matrix(np.argmax(label_test, axis=1), predicted))

predicted = np.array(predicted)

pred = []
actu = []
for p in predicted:
    pre_card = label_card[p]
    pred.append(pre_card)
for actual in label_test2:
    actual = label_card[actual]
    actu.append(actual)

for idx in range(len(label_test)):
    pre = pred[idx]
    act = actu[idx]
    print('予想：',pre,'　　　','正解ラベル：',act)
