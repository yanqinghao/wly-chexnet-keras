# coding=utf-8
from __future__ import absolute_import, print_function

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder,String
import os
from model import ModelFactory
import numpy as np
import json
from suanpan.storage import storage
import shutil
import time

# 定义输入
@dc.input(Folder(key="inputData1", required=True))
@dc.input(Folder(key="inputData2", required=True))
@dc.param(String(key="param1"))
@dc.param(String(key="param2"))
# 定义输出
@dc.output(Folder(key="outputData1", required=True))
def Demo(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData1 数据
    print("**********************start****************************")
    print(time.ctime())
    print(args.inputData1)
    print(args.param1)
    print(args.param2)
    if args.param1==None:
        input_image = os.path.join(args.inputData1, "predict-image")
    else:
        input_image = "./imagefile"
        storage.downloadFolder(folderName=args.param1, folderPath=input_image)

    if args.param2==None:
        input_model = os.path.join(args.inputData2, "model")
    else:
        input_model = "./modelfile"
        storage.downloadFolder(folderName=args.param2, folderPath=input_model)

    test_path = input_image
    test_file = os.listdir(test_path)
    output_weights_path = os.path.join(input_model, "weight.h5")
    print(output_weights_path)

    with open(os.path.join(input_model, "label_map.json"), "r") as load_f:
        label_map = json.load(load_f)
        print(label_map)
    class_names = list(label_map.keys())
    print(class_names)
    images = image.load_img(os.path.join(
        test_path, test_file[0]), grayscale=False, target_size=(224, 224))
    x = image.img_to_array(images) / 255.0
    x = np.expand_dims(x, axis=0)

    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name="DenseNet121", class_names=class_names, use_base_weights=False, weights_path=output_weights_path
    )
    print(model.summary())
    print(len(model.layers))

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
        # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
    else:
        model_train = model
    print(time.ctime())
    print("**********************predict****************************")
    result = model_train.predict(x)
    print(time.ctime())
    print(result)
    if len(class_names)==2:
        result = class_names[1] if result > 0.5 else class_names[0]
    else:
        result = class_names[np.argsort(result[0])[::-1][0]]
    print(result)
    print(os.path.join(args.outputData1, "result.json"))
    # 自定义代码
    with open(os.path.join(args.outputData1, test_file[0][:-4]+".json"), "w") as f:
        json.dump({"result": result}, f)
        print("save...")
    if args.param1!=None:
        shutil.rmtree(input_image)
    if args.param2!=None:
        shutil.rmtree(input_model)
    # 将 args.outputData1 作为输出发送给下一节点
    return args.outputData1


if __name__ == "__main__":
    Demo()
