import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf
import random

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility

plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# set_session(tf.Session(config=config))
# some constants
NUM_CLASSES = 21  #もともと4
input_shape = (300, 300, 3)

priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb') )
bbox_util = BBoxUtility(NUM_CLASSES, priors)

#  keysにはfilenameに書かれてある画像名が出力される！

# 変更箇所二つのうち一つ目
#################################################################################
#  ここにXML_pkl.pyで保存したものを入れる!!!!
gt = pickle.load(open('/Users/user/SSD/12_moji_menkyo.pkl', 'rb'))
##################################################################################




keys = sorted(gt.keys())
print(keys)



num_train =( len(keys) - 2)


#  ランダムに学習用とテスト用に分ける
list_idx = list(range(0,len(keys)))
train_idx = random.sample(list_idx, num_train)
print(train_idx)
test_idx = set(list_idx) - set(train_idx)
test_idx = list(test_idx)
print(test_idx)

#  全てtrain_keysに入れてval_keysを空にすると学習が止まる
train_keys = []
val_keys = []
for tr_idx in train_idx:
    train_keys.append(keys[tr_idx])
for vl_idx in test_idx:
    val_keys.append(keys[vl_idx])


#train_keys = keys[:num_train]
print('train_keys',train_keys)
print('train_keysの数：',len(train_keys))
#val_keys = keys[num_train:]
print('val_keys',val_keys)
#num_val = len(val_keys)


class Generator(object):
    def __init__(self, gt, bbox_util,
                 batch_size, path_prefix,
                 train_keys, val_keys, image_size,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 hflip_prob=0.5,
                 vflip_prob=0.5,
                 do_crop=True,
                 crop_area_range=[0.75, 1.0],
                 aspect_ratio_range=[3./4., 4./3.]):
        self.gt = gt
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.train_keys = train_keys
        self.val_keys = val_keys
        self.train_batches = len(train_keys)
        self.val_batches = len(val_keys)
        self.image_size = image_size
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.lighting_std = lighting_std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.do_crop = do_crop
        self.crop_area_range = crop_area_range
        self.aspect_ratio_range = aspect_ratio_range
        
    def grayscale(self, rgb):
        return rgb.dot([0.299, 0.587, 0.114])

    def saturation(self, rgb):
        gs = self.grayscale(rgb)
        alpha = 2 * np.random.random() * self.saturation_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha + (1 - alpha) * gs[:, :, None]
        return np.clip(rgb, 0, 255)

    def brightness(self, rgb):
        alpha = 2 * np.random.random() * self.brightness_var 
        alpha += 1 - self.saturation_var
        rgb = rgb * alpha
        return np.clip(rgb, 0, 255)

    def contrast(self, rgb):
        gs = self.grayscale(rgb).mean() * np.ones_like(rgb)
        alpha = 2 * np.random.random() * self.contrast_var 
        alpha += 1 - self.contrast_var
        rgb = rgb * alpha + (1 - alpha) * gs
        return np.clip(rgb, 0, 255)

    def lighting(self, img):
        cov = np.cov(img.reshape(-1, 3) / 255.0, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigvec.dot(eigval * noise) * 255
        img += noise
        return np.clip(img, 0, 255)
    
    def horizontal_flip(self, img, y):
        if np.random.random() < self.hflip_prob:
            img = img[:, ::-1]
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y
    
    def vertical_flip(self, img, y):
        if np.random.random() < self.vflip_prob:
            img = img[::-1]
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y
    
    def random_sized_crop(self, img, targets):
        img_w = img.shape[1]
        img_h = img.shape[0]
        img_area = img_w * img_h
        random_scale = np.random.random()
        random_scale *= (self.crop_area_range[1] -
                         self.crop_area_range[0])
        random_scale += self.crop_area_range[0]
        target_area = random_scale * img_area
        random_ratio = np.random.random()
        random_ratio *= (self.aspect_ratio_range[1] -
                         self.aspect_ratio_range[0])
        random_ratio += self.aspect_ratio_range[0]
        w = np.round(np.sqrt(target_area * random_ratio))     
        h = np.round(np.sqrt(target_area / random_ratio))
        if np.random.random() < 0.5:
            w, h = h, w
        w = min(w, img_w)
        w_rel = w / img_w
        w = int(w)
        h = min(h, img_h)
        h_rel = h / img_h
        h = int(h)
        x = np.random.random() * (img_w - w)
        x_rel = x / img_w
        x = int(x)
        y = np.random.random() * (img_h - h)
        y_rel = y / img_h
        y = int(y)
        img = img[y:y+h, x:x+w]
        new_targets = []
        for box in targets:
            cx = 0.5 * (box[0] + box[2])
            cy = 0.5 * (box[1] + box[3])
            if (x_rel < cx < x_rel + w_rel and
                y_rel < cy < y_rel + h_rel):
                xmin = (box[0] - x_rel) / w_rel
                ymin = (box[1] - y_rel) / h_rel
                xmax = (box[2] - x_rel) / w_rel
                ymax = (box[3] - y_rel) / h_rel
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(1, xmax)
                ymax = min(1, ymax)
                box[:4] = [xmin, ymin, xmax, ymax]
                new_targets.append(box)
        new_targets = np.asarray(new_targets).reshape(-1, targets.shape[1])
        return img, new_targets
    
    #  generateがこのクラスのメインであり中核。他はグレースケールやら輝度やらこの関数を構築するためのパーツ
    #  self.bbox_util.assign_boxes(y)で一番重なりが大きいバウンディングボックスの情報を取り出してる　＝　 y　→targetsに追加
    def generate(self, train=True):
        while True:
            if train:
                shuffle(self.train_keys)
                keys = self.train_keys
            else:
                shuffle(self.val_keys)
                keys = self.val_keys
            inputs = []
            targets = []
            for key in keys:            
                img_path = self.path_prefix + key
                img = imread(img_path).astype('float32')
                y = self.gt[key].copy()
                if train and self.do_crop:
                    img, y = self.random_sized_crop(img, y)
                img = imresize(img, self.image_size).astype('float32')
                if train:
                    shuffle(self.color_jitter)
                    for jitter in self.color_jitter:
                        img = jitter(img)
                    if self.lighting_std:
                        img = self.lighting(img)
                    if self.hflip_prob > 0:
                        img, y = self.horizontal_flip(img, y)
                    if self.vflip_prob > 0:
                        img, y = self.vertical_flip(img, y)
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)                
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    #print(tmp_targets.shape)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets


# 変更箇所二つのうち二つ目。あとは実行すれば学習できる
#################################################################################
#  ここにlabelImgでボックスを作成した画像が入っているディレクトリを入力

# 上のpklファイルを入れるのとここの画像ファイルを指定すれば学習できる。ただしpklの元になるxmlで指定した画像と以下で指定した画像が一致しなければならない
path_prefix = '/Users/user/Desktop/12moji/images/'

#################################################################################

gen = Generator(gt, bbox_util, 2, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)


model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)


freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False
        
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule)]


base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)


nb_epoch = 30

history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(True),
                              nb_val_samples=gen.train_batches,
                              nb_worker=1
                               #validation_steps=1
                              )


inputs = []
images = []
img_path = '/Users/user/SSD/train_menkyo_images/menkyosyou.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = '/Users/user/SSD/000001.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path =  '/Users/user/SSD/train_menkyo_images/menkyo3.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
img_path = '/Users/user/SSD/train_menkyo_images_full/menkyo_fuku15.jpg'
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))
inputs.shape



preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)


#  imagesは検知させたい画像でiはその画像に番号振ったもの
zahyou = []
for num, img in enumerate(images):
    # Parse the outputs.
    #  これは予測したもの全てのそれぞれの値が出てしまう！！ラベルである確率が低いものまで全て入ってしまってる
    det_label = results[num][:, 0]
    #  det_confは確率 softmax使ってるから。そのラベルである確率
    det_conf = results[num][:, 1]
    det_xmin = results[num][:, 2]
    det_ymin = results[num][:, 3]
    det_xmax = results[num][:, 4]
    det_ymax = results[num][:, 5]
    

    # Get detections with confidence higher than 0.6.
    #  確率が６０％以上のものだけのインデックス番号を取り出す
    # →６０％以上の[label, confidence, xmin, ymin, xmax, ymax]＝top_indicesを取り出す
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    
    if len(top_indices) <= 50:
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.5]
    
    
    if len(top_indices) <= 50:
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.4]
        
    if len(top_indices) <= 50:
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.3]
          
        
    if len(top_indices) <= 50:
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.23]
    
    #if len(top_indices) <= 50:
    #    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.18]
        
    

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

#  top_confで６０％以上のラベルであろう数字が一番確率が高い順から入ってる！！top_conf.shape[0]は見つけだした枠の数の分だけの確率が入ってる。
# ６０％以上で検知できた物体が４つあるなら４とでる
# top_conf.shape[0]には検知した物体の数＝枠の数の分だけの確率が格納されてる
    for i in range(top_conf.shape[0]):
        
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        # xmax ymaxが右下の座標
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        
     
        zahyou.append({'{}枚目'.format(num+1): [(xmin,ymin),(xmax , ymin ) ,(xmin, ymax) ,(xmax ,ymax)] })
        score = top_conf[i]
        label_name = int(top_label_indices[i])
        #label_name = voc_classes[label - 1]
        #  表示する文字→確率とラベルの名前catとか
        #display_txt = '{:0.2f}, {}'.format(score, label_name)
        display_txt=None
        
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        zahyou =  [ (xmin, ymin), (xmax,ymax) ]
        color = colors[label_name]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        
        print('座標：',zahyou)
      
    

    