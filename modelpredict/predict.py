from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
from model import ModelFactory
import json

test_dir = "../cats_and_dogs_small/test"
output_weights_path = "./weight.h5"
class_names = ["dog", "cat"]

with open("./record.json", "r") as load_f:
    label_map = json.load(load_f)
    print(label_map)

model_factory = ModelFactory()
model = model_factory.get_model(
    class_names, use_base_weights=False, weights_path="./weight.h5"
)
print(model.summary())
print(len(model.layers))

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32
)

gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
if gpus > 1:
    print(f"** multi_gpu_model is used! gpus={gpus} **")
    model_train = multi_gpu_model(model, gpus)
    # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
else:
    model_train = model
result = model_train.predict_generator(test_generator, len(test_generator))
