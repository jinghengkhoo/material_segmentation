import os
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications.convnext import ConvNeXtBase

from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

base_model = ConvNeXtBase(include_top=False, weights='imagenet',  input_shape = (224,224,3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)       
x = layers.Dense(128, activation='relu')(x)       
out = layers.Dense(21, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)

for layer in base_model.layers:
  layer.trainable = False

model = Model(inputs=base_model.input, outputs=out)
model.load_weights('wood')

model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=[
              keras.metrics.Precision(name='precision'),
              keras.metrics.Recall(name='recall'),
              keras.metrics.AUC(name='auc'),
              tfa.metrics.F1Score(name='f1_score', num_classes = 21)
              ]) 

model.save("model.savedmodel")