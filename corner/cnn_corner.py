from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,random
from sklearn import cross_validation
from cre_image_label import image_point

#バッチサイズ
batch_size = 1

# エポック数
epoch = 3


# 画像を変形させて、頂点の座標も移動させる関数
#image_train,image_testには画像
# label_train,label_testには頂点の座標（左上、左下、右上、右下）が格納されている
image_train, image_test, label_train, label_test = image_point(10 , 60 , 360 ,50 ,50 ,5 ,8 ,5 , 189, 194, 8, 194, 189)




image_train = np.array(image_train)
image_test = np.array(image_test)

num_train = len(label_train)
num_test = len(label_test)


LT_x=[]
LT_y=[]
LB_x=[]
LB_y=[]
RT_x=[]
RT_y=[]
RB_x=[]
RB_y=[]

#  座標を全部バラバラにする処理
for n in range(num_train):
    
    lbl_train = label_train[n]

    LT_x.append(lbl_train[0][0])
    LT_y.append(lbl_train[0][1])

    LB_x.append(lbl_train[1][0])
    LB_y.append(lbl_train[1][1])

    RT_x.append(lbl_train[2][0])
    RT_y.append(lbl_train[2][1])

    RB_x.append(lbl_train[3][0])
    RB_y.append(lbl_train[3][1])


LT_x = np.array(LT_x)
LT_y = np.array(LT_y)
LB_x = np.array(LB_x)
LB_y = np.array(LB_y)
RT_x = np.array(RT_x)
RT_y = np.array(RT_y)
RB_x = np.array(RB_x)
RB_y = np.array(RB_y)


test_LT_x=[]
test_LT_y=[]
test_LB_x=[]
test_LB_y=[]
test_RT_x=[]
test_RT_y=[]
test_RB_x=[]
test_RB_y=[]

for n in range(num_test):
    
    lbl_test = label_test[n]
    
    test_LT_x.append(lbl_test[0][0])
    test_LT_y.append(lbl_test[0][1])

    test_LB_x.append(lbl_test[1][0])
    test_LB_y.append(lbl_test[1][1])

    test_RT_x.append(lbl_test[2][0])
    test_RT_y.append(lbl_test[2][1])

    test_RB_x.append(lbl_test[3][0])
    test_RB_y.append(lbl_test[3][1])


test_LT_x = np.array(test_LT_x)
test_LT_y = np.array(test_LT_y)
test_LB_x = np.array(test_LB_x)
test_LB_y = np.array(test_LB_y)
test_RT_x = np.array(test_RT_x)
test_RT_y = np.array(test_RT_y)
test_RB_x = np.array(test_RB_x)
test_RB_y = np.array(test_RB_y)


#  正規化
image_train = image_train.astype("float") / 255
image_test  = image_test.astype("float") / 255



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

    LT_x = Dense(1,activation='linear',name='LT_x')(x)
    LT_y = Dense(1,activation='linear',name='LT_y')(x)
    LB_x = Dense(1,activation='linear',name='LB_x')(x)
    LB_y = Dense(1,activation='linear',name='LB_y')(x)
    RT_x = Dense(1,activation='linear',name='RT_x')(x)
    RT_y = Dense(1,activation='linear',name='RT_y')(x)
    RB_x = Dense(1,activation='linear',name='RB_x')(x)
    RB_y = Dense(1,activation='linear',name='RB_y')(x)


    # 上の流れをインスタンス化→コンパイルするために
    multiModel = Model(inputs= i , outputs = [LT_x, LT_y, LB_x, LB_y, RT_x, RT_y, RB_x, RB_y])

    multiModel.compile(loss={'LT_x': 'mean_squared_error',
                        'LT_y': 'mean_squared_error',
                        'LB_x': 'mean_squared_error',
                        'LB_y': 'mean_squared_error',
                        'RT_x': 'mean_squared_error',
                        'RT_y': 'mean_squared_error',
                        'RB_x': 'mean_squared_error',
                        'RB_y': 'mean_squared_error'},
                   optimizer='Adam')


    return multiModel


model = build_model(image_train.shape[1:])





#label_train_size
model.fit(image_train,  {'LT_x': LT_x,
                    'LT_y': LT_y,
                    'LB_x': LB_x,
                    'LB_y': LB_y,
                    'RT_x': RT_x,
                    'RT_y': RT_y,
                    'RB_x': RB_x,
                   'RB_y': RB_y}, batch_size = batch_size, nb_epoch = epoch)

#予測結果（％）
pre_LT_x , pre_LT_y , pre_LB_x , pre_LB_y , pre_RT_x , pre_RT_y , pre_RB_x , pre_RB_y = model.predict(image_test)



# 以下は予測した座標（％）をピクセルに変換し、正解ラベルと並べて表示する処理

x = image_train.shape[1]
y = image_train.shape[2]



# ％からピクセルへ座標変換 →四捨五入→整数に直す

print('\n左上の座標(x,y)',' ', '左上座標の予想(x,y)', sep='\t')   
for num in range(len(pre_LT_x)):
    LT=[] 
    LT_x = np.round(pre_LT_x * x)
    LT.append( int(LT_x[num][0]) )
    
    LT_y = np.round(pre_LT_y * y) 
    LT.append(int(LT_y[num][0]))
    
    label_LT = []
    label_LT_x = np.round(test_LT_x * x )
    label_LT.append(int(label_LT_x[num]))
    
    label_LT_y = np.round(test_LT_y * y )
    label_LT.append(int(label_LT_y[num]))
    
    print(label_LT,'　　　　　',LT)


    
    
print('\n左下の座標(x,y)',' ', '左下座標の予想(x,y)', sep='\t')   
for num in range(len(pre_LB_x)):    
    LB=[] 
    LB_x = np.round(pre_LB_x * x )
    LB.append(int(LB_x[num][0]))
    
    LB_y = np.round(pre_LB_y * y )
    LB.append(int(LB_y[num][0]))
    
    label_LB = []
    label_LB_x = np.round(test_LB_x * x )
    label_LB.append(int(label_LB_x[num]))
    
    label_LB_y = np.round(test_LB_y * y )
    label_LB.append(int(label_LB_y[num]))
    
    print(label_LB,'　　　　　',LB)
    
    
    
RT=[]    
print('\n右上の座標(x,y)',' ', '右上座標の予想(x,y)', sep='\t')   
for num in range(len(pre_RT_x)):
    RT=[] 
    RT_x = np.round(pre_RT_x * x )
    RT.append(int(RT_x[num][0]))
    
    RT_y = np.round(pre_RT_y * y )
    RT.append(int(RT_y[num][0]))
    
    label_RT = []
    label_RT_x = np.round(test_RT_x * x )
    label_RT.append(int(label_RT_x[num]))
    
    label_RT_y = np.round(test_RT_y * y )
    label_RT.append(int(label_RT_y[num]))
    
    print(label_RT,'　　　　　',RT)


    
    
print('\n右下の座標(x,y)',' ', '右下座標の予想(x,y)', sep='\t')   
for num in range(len(pre_RB_x)):    
    RB=[] 
    RB_x = np.round(pre_RB_x * x )
    RB.append(int(RB_x[num][0]))
    
    RB_y = np.round(pre_RB_y * y ) 
    RB.append(int(RB_y[num][0]))
    
    label_RB = []
    label_RB_x = np.round(test_RB_x * x )
    label_RB.append(int(label_RB_x[num]))
    
    label_RB_y = np.round(test_RB_y * y )
    label_RB.append(int(label_RB_y[num]))
    
    print(label_RB,'　　　　　　',RB)
    
    

