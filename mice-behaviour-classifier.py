import os
import glob
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder


import tensorflow as tf

from tensorflow.keras.layers import Input,GlobalAveragePooling2D,GlobalMaxPooling2D,Dense



dirs=glob.glob(os.getcwd()+'/*')
#dirs = ['/hpf/projects/prescott/classifier_latency/neural_network/chris', '/hpf/projects/prescott/classifier_latency/neural_network/mehdi']
# dirs = ['C:\\Users\\MEHDI AZADGOLEH\OneDrive - SickKids\\Mehdi\\Behavioural_Tests\\classifier\\neural network\\mehdi']
sub_dirs=[]

print('directories: ', dirs)

for i in dirs:
    sub_dirs.append(glob.glob(i+'/*'))

print('sub directories: ', sub_dirs)


labels=[]
images=[]

for d in range(len(sub_dirs)):
    for sd in sub_dirs[d]:
        label=sd.split('_')[-1]
        path_to_images=glob.glob(sd+'/*')
        for path in path_to_images:
            labels.append(label)
            images.append(cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE),cv2.COLOR_GRAY2RGB))
            

labels=np.array(labels)
labels=labels.reshape(-1,1)
images=np.array(images)

labels[labels=='slow']=0
labels[labels=='fast']=1

enc=OneHotEncoder(sparse=False)

enc.fit(labels)

onehot_labels=enc.transform(labels)

portion=0.80

idx = np.random.permutation(len(images))
images, onehot_labels = images[idx], onehot_labels[idx]



height = images[0].shape[0]
width = images[0].shape[1]
channels = images[0].shape[2] 

images_ = np.ndarray(shape=(len(images), height, width, channels), dtype = np.float32)


preprocess=tf.keras.applications.mobilenet_v2.preprocess_input

# loop through all images
for i in range(len(images)):
      
      img = images[i]
      frame_height = img.shape[0]
      frame_width = img.shape[1]
      print("resolusiton is: {} x {}".format(frame_height, frame_width) )
      img = preprocess(img)
      images_[i] = img

train_labels=onehot_labels[0:int(portion*len(labels))]
train_image=images_[0:int(portion*len(labels))]

test_labels=onehot_labels[int(portion*len(labels)):]
test_image=images_[int(portion*len(labels)):]


base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(500,1008,3), alpha=1.0, include_top=False, weights='imagenet',
    input_tensor=None, pooling=None,
    classifier_activation='softmax')

# base_model = tf.keras.applications.resnet50.ResNet50(
#     include_top=False, weights='imagenet', input_tensor=None,
#     input_shape=(500,1008,3), pooling=None, classes=1000, **kwargs
# )


base_model.summary()

inputs = Input(shape=(500,1008,3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = GlobalMaxPooling2D()(x)

x=Dense(512,activation='relu')(x)
# A Dense classifier with a single unit (binary classification)
outputs = Dense(2,activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()


for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))


# # training the last two blocks
for layer in base_model.layers[:134]:
    layer.trainable = False
for layer in model.layers[134:]:
    layer.trainable = True

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0001),
             loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
   metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(train_image,train_labels, validation_data=(test_image,test_labels), batch_size=16, epochs=10000)