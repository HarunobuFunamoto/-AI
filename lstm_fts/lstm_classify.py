import pandas as pd
import numpy as np
from sklearn import cross_validation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from keras.layers import Dense,LSTM, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adadelta,Adam


epoch = 30

# batch_size , neurons, neurons2, neurons3, embedding_vecor_length, learning_rateはエポック３０での試行錯誤の末、一番良かった組み合わせにしています
batch_size = 2

# 出力する時の次元数。
# LSTM１層目
neurons = 70
# LSTM２層目
neurons2 = 70
# LSTM３層目
neurons3 = 100

# 出力する密ベクトルの次元数
embedding_vecor_length = 2

# アダムの学習率
learning_rate = 0.5




data_csv = pd.read_csv('menkyo_info.csv',header=None)
data_csv.columns = ['moji','label']
moji = data_csv['moji']
#moji_array = np.array(moji)
moji_list = moji.tolist()

label = data_csv['label']
#label_array = np.array(label)
#print(label_array)
lbl_list = label.tolist()
label_li = []
for lbl in lbl_list:
    for i in range(5):
        label_li.append(lbl)



moji_str =  ''
for mojis in moji_list:
    moji_str += mojis

tokenizer = Tokenizer()
tokenizer.fit_on_texts(moji_str)
moji_seq = tokenizer.texts_to_sequences(moji_str)
vocab_size = len(tokenizer.word_index) + 1


# load_ssd_predict2.pyでSSDで予測した座標が格納されていることを想定してnpyファイルをロード。
point = np.load('zahyou.npy')
#print(point)
xmin = []
ymin = []
xmax = []
ymax = []
for p in point:
    xmin.append(p[0])
    ymin.append(p[1])
    xmax.append(p[2])
    ymax.append(p[3])
    
#print(len(xmin_zahyou))

# 文字と座標は独立していて良い！！それぞれ一つ一つにラベルが振ってあれば問題なし
#data=[文字列、座標]
data = []
#moji_zahyou = []
for idx in range(len(xmin)):
    
    data.append(moji_seq[idx])
    
    data.append([xmin[idx]])

    data.append([ymin[idx]])
    
    data.append([xmax[idx]])
    
    data.append([ymax[idx]])
    
    #data.append(moji_zahyou)
#print('data',len(data))
#print('xmins',len(xmins))
#print(data)


count_max = []
#moji_zahyou = []
for idx in range(len(xmin)):
    
    count_max.append(moji_seq[idx][0])
    
    count_max.append(xmin[idx])

    count_max.append(ymin[idx])
    
    count_max.append(xmax[idx])
    
    count_max.append(ymax[idx])
    
max_num = max(count_max)





#moji_train = np.array(moji_train)
#moji_test = np.array(moji_test)
#label_train = np.array(label_train)
#label_test = np.array(label_test)


labels = list(set(label_li))
label_dic = {}
for idx, lbl in enumerate(labels):
     label_dic[idx] = lbl



nb_label = len(labels)
#print(nb_label)
#print(label_train)

label_list =[]
for label in label_li:
    for key , val in label_dic.items():
        if val == label:
            label_list.append(key)
        else:continue
        
    
moji_train, moji_test, label_train, label_test = cross_validation.train_test_split(data, label_list)


moji_train = np.array(moji_train)
moji_test = np.array(moji_test)
label_train2 = np.array(label_train)
label_test2 = np.array(label_test)


for train in moji_train:
    num = len(train)


#reshape(データセットの数＝LSTMのタイムステップ、一度に入力させたいデータの数＝データを何個使って予測をさせるか、データ１種類につき１個のデータだから１)
moji_tra = moji_train.reshape( len(moji_train) , num)
moji_tes = moji_test.reshape( len(moji_test) , num)
label_train = np_utils.to_categorical(label_train2, nb_label)
label_test = np_utils.to_categorical(label_test2, nb_label)






#This is the size of the vocabulary in the text data. 
num_word = max_num +1


#input_length argumetは各入力シーケンスのサイズを決定
#[[2 3 1 2 1]
# [3 3 0 2 3]]が入力なら　input_length =５
# いじらなくて良い！！
input_length = 1

def build_model(train_data , neurons ,neurons2,neurons3, batich_size):
    model = Sequential()
    model.add(Embedding(num_word,embedding_vecor_length,input_length = input_length))
    model.add(Conv1D(32, 3, border_mode='same', activation='relu'))
    model.add(LSTM(neurons, batch_input_shape= (batch_size, train_data.shape[1]), activation='relu',return_sequences = True ))
    model.add(LSTM(neurons2,return_sequences = True ))
    model.add(LSTM(neurons3,return_sequences = False ))
    model.add(Dense(7, activation='softmax'))
    Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


model = build_model(moji_tra , neurons , neurons2 , neurons3 ,1)


model.fit(moji_tra, label_train, epochs=epoch ,batch_size=batch_size,validation_data=(moji_tes, label_test))
model.reset_states()


#  改善案
predicted = model.predict(moji_seq , batch_size=batch_size)
model.reset_states()


#  免許証で出力した順に
answer = []
for pre in predicted:
    for idx , num in enumerate(pre):
        if num == np.max(pre):
            ans = label_dic[idx]
            answer.append(ans)

label = []
for lbl_test in label_list:
    lbl = label_dic[lbl_test]
    label.append(lbl)
            
print('\n正解',' 　　　　', '予想', sep='\t')   



for idx in range(len(lbl_list)):
    
    print(lbl_list[idx],'　　',answer[idx],sep='\t')