from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
from model import ModelFactory
import json
from keras.preprocessing import image
import numpy as np

test_file = "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\cats_and_dogs_small\\test\\cats\\cat.1500.jpg"
outputWeightsPath = "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\modelpredict\\weight.h5"

with open("label_map.json", "r") as load_f:
    labelMap = json.load(load_f)
    print(labelMap)
classNames = labelMap.keys()

images = image.load_img(test_file, grayscale=False, target_size=(224, 224))
x = image.img_to_array(images) / 255.0
x = np.expand_dims(x, axis=0)

model_factory = ModelFactory()
model = model_factory.get_model(
    classNames, use_base_weights=False, weights_path=outputWeightsPath
)
print(model.summary())
print(len(model.layers))

model_train = model

result = model_train.predict(x)
print(result)
if len(result) == 1:
    result = classNames[1] if result > 0.5 else classNames[0]
else:
    result = classNames[np.argsort(result[0])[::-1][0]]
print(result)
