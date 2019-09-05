# coding=utf-8
from __future__ import absolute_import, print_function
from __suanpan__ import tensorflow

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, String, Float
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
import os
from weights import get_class_weights
from utility import get_sample_counts
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from callbackrealtime import CSVLoggerOss
from model import ModelFactory
import json
from suanpan.storage import storage
import pandas as pd
import shutil
import time
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


# 定义输入
@dc.input(Folder(key="inputData1", required=True))
@dc.param(String(key="param1"))
@dc.param(Float(key="param2"))
# 定义输出
@dc.output(Folder(key="outputData1", required=True))
def Demo(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData1 数据
    print(args.inputData1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print("GPU device:", tf.test.gpu_device_name())
    print("is GPU available:", tf.test.is_gpu_available())
    print("list all devices:", device_lib.list_local_devices())
    print("check all devices:", K.tensorflow_backend._get_available_gpus())

    downloadPath = "/tmp/images"
    imagePath = "/tmp/images_sort"
    startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    timeStamp = time.time()
    endTime = None
    useTime = None
    imageNum = None
    trainingInfo = {
        "startTime": startTime,
        "endTime": endTime,
        "useTime": useTime,
        "imageNum": imageNum,
    }

    if not os.path.exists(
        "/root/.keras/models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ):
        os.makedirs("/root/.keras/models")
        if os.path.exists(
            "/wly/model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
        ):
            print("******************Load local model******************")
            shutil.copyfile(
                "/wly/model/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
                "/root/.keras/models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
            )
        else:
            storage.download(
                "common/model/keras/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
                "/root/.keras/models/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
            )
    # 自定义代码
    outputDir = args.outputData1
    if os.path.exists(downloadPath):
        shutil.rmtree(downloadPath)
    if os.path.exists(imagePath):
        shutil.rmtree(imagePath)

    if args.param1 == None:
        trainDir = os.path.join(args.inputData1, "train")
        validationDir = os.path.join(args.inputData1, "test")
    else:
        ossPath = args.param1
        with open("training_log.json", "w") as f:
            json.dump({"info": "downloadmodel", "data": None}, f)
        if os.path.split(ossPath)[0] != None:
            storage.upload(
                "{}/training_log.json".format(os.path.split(ossPath)[0]),
                "training_log.json",
            )
        trainTestSplit = args.param2 if args.param2 != None else 0.8
        print("************Split train and test : {}***********".format(trainTestSplit))
        with open("training_log.json", "w") as f:
            json.dump({"info": "downloadimage", "data": None}, f)
        if os.path.split(ossPath)[0] != None:
            storage.upload(
                "{}/training_log.json".format(os.path.split(ossPath)[0]),
                "training_log.json",
            )
        storage.download(ossPath, downloadPath)
        print(os.path.split(ossPath)[0] + "/detail.json")
        storage.download(os.path.split(ossPath)[0] + "/detail.json", "detail.json")
        with open("detail.json", "r") as f:
            label_detail = json.load(f)
        trainingInfo["imageNum"] = len(label_detail["images"])

        os.mkdir(imagePath)
        os.mkdir(os.path.join(imagePath, "train"))
        os.mkdir(os.path.join(imagePath, "test"))
        labels = {}
        labels["name"] = []
        labels["level"] = []
        for i in label_detail["images"]:
            if "name" in list(i.keys()) and "level" in list(i.keys()):
                labels["name"].append(i["name"])
                labels["level"].append(i["level"])
        labels = pd.DataFrame(labels)
        for i in list(set(labels["level"].values)):
            os.mkdir(os.path.join(imagePath, "train", i))
            os.mkdir(os.path.join(imagePath, "test", i))
        print(list(set(labels["level"].values)))
        foldercheck = []
        with open("training_log.json", "w") as f:
            json.dump({"info": "imageformat", "data": None}, f)
        if os.path.split(ossPath)[0] != None:
            storage.upload(
                "{}/training_log.json".format(os.path.split(ossPath)[0]),
                "training_log.json",
            )
        for m in list(set(labels["level"].values)):
            labels_tmp = labels[labels["level"] == m]
            foldercheck.append(len(labels_tmp) * trainTestSplit > (len(labels_tmp) - 1))
            print(labels_tmp["level"])
            for j in range(len(labels_tmp)):
                if j < len(labels_tmp) * trainTestSplit:
                    print(os.path.join(imagePath, "train", labels_tmp["level"].iloc[j]))
                    print(os.path.join(downloadPath, labels_tmp["name"].iloc[j]))
                    src = os.path.join(
                        downloadPath, labels_tmp["name"].iloc[j] + ".png"
                    )
                    dst = os.path.join(imagePath, "train", labels_tmp["level"].iloc[j])
                    shutil.copy(src, os.path.join(dst, str(j) + os.path.split(src)[1]))
                else:
                    src = os.path.join(
                        downloadPath, labels_tmp["name"].iloc[j] + ".png"
                    )
                    dst = os.path.join(imagePath, "test", labels_tmp["level"].iloc[j])
                    shutil.copy(src, os.path.join(dst, str(j) + os.path.split(src)[1]))
        trainDir = os.path.join(imagePath, "train")
        if any(foldercheck):
            validationDir = trainDir
        else:
            validationDir = os.path.join(imagePath, "test")

    outputWeightsPath = os.path.join(outputDir, "weight.h5")

    imageSize = 224
    batchSize = 32
    classNum = len(os.listdir(trainDir))
    Epoch = 10
    learningRate = 0.001
    inputShape = (224, 224, 3)

    with open("training_log.json", "w") as f:
        json.dump({"info": "loadmodel", "data": None}, f)
    if os.path.split(ossPath)[0] != None:
        storage.upload(
            "{}/training_log.json".format(os.path.split(ossPath)[0]),
            "training_log.json",
        )

    models = ModelFactory()

    trainDatagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        horizontal_flip=True,
        vertical_flip=False,
        height_shift_range=0.05,
        width_shift_range=0.1,
        rotation_range=5,
        shear_range=0.1,
        fill_mode="reflect",
        zoom_range=0.15,
        rescale=1.0 / 255,
    )
    trainGenerator = (
        trainDatagen.flow_from_directory(
            trainDir,
            target_size=(imageSize, imageSize),
            batch_size=batchSize,
            class_mode="categorical",
        )
        if classNum > 2
        else trainDatagen.flow_from_directory(
            trainDir,
            target_size=(imageSize, imageSize),
            batch_size=batchSize,
            class_mode="binary",
        )
    )

    labelMap = trainGenerator.class_indices
    classNames = list(labelMap.keys())
    with open(os.path.join(outputDir, "label_map.json"), "w") as f:
        json.dump(labelMap, f)
        print("save label map...")

    # train_counts, train_pos_counts = get_sample_counts(trainDir, "train", classNames)
    print("** compute class weights from training data **")
    # classWeights = get_class_weights(train_counts, train_pos_counts, multiply=1)
    print("** class_weights **")

    validationDatagen = ImageDataGenerator(rescale=1.0 / 255)
    validationGenerator = (
        validationDatagen.flow_from_directory(
            validationDir,
            target_size=(imageSize, imageSize),
            batch_size=batchSize,
            class_mode="categorical",
        )
        if classNum > 2
        else validationDatagen.flow_from_directory(
            validationDir,
            target_size=(imageSize, imageSize),
            batch_size=batchSize,
            class_mode="binary",
        )
    )

    model = models.get_model(
        model_name="DenseNet121", class_names=classNames, input_shape=inputShape
    )
    print(model.summary())
    print(len(model.layers))

    gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
    if gpus > 1:
        print(f"** multi_gpu_model is used! gpus={gpus} **")
        model_train = multi_gpu_model(model, gpus)
        # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
        checkpoint = MultiGPUModelCheckpoint(
            filepath=outputWeightsPath, base_model=model, save_best_only=True, verbose=1
        )
    else:
        model_train = model
        checkpoint = ModelCheckpoint(
            outputWeightsPath, monitor="val_acc", save_best_only=True, verbose=1
        )

    optimizer = Adam(lr=learningRate)
    if len(classNames) > 2:
        model_train.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"]
        )
    else:
        model_train.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["acc"]
        )
    if args.param1 == None:
        csvlog = CSVLogger(os.path.join(args.outputData1, "training_log.csv"))
    else:
        csvlog = CSVLoggerOss("training_log.csv", osspath=os.path.split(args.param1)[0])
    callbacks = [
        checkpoint,
        # TensorBoard(log_dir=os.path.join(outputDir, "logs"), batch_size=32),
        ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.2,
            patience=5,
            verbose=1,
            mode="max",
            min_lr=1e-8,
        ),
        csvlog,
    ]

    with open("training_log.json", "w") as f:
        json.dump({"info": "trainmodel", "data": None}, f)
    if os.path.split(ossPath)[0] != None:
        storage.upload(
            "{}/training_log.json".format(os.path.split(ossPath)[0]),
            "training_log.json",
        )

    history = (
        model_train.fit_generator(
            generator=trainGenerator,
            steps_per_epoch=len(trainGenerator),
            epochs=Epoch,
            validation_data=validationGenerator,
            validation_steps=len(validationGenerator),
            callbacks=callbacks,
            # class_weight=classWeights,
            workers=8,
            shuffle=False,
        )
        if len(classNames) > 2
        else model_train.fit_generator(
            generator=trainGenerator,
            steps_per_epoch=len(trainGenerator),
            epochs=Epoch,
            validation_data=validationGenerator,
            validation_steps=len(validationGenerator),
            callbacks=callbacks,
            workers=8,
            shuffle=False,
        )
    )
    if os.path.exists(downloadPath):
        shutil.rmtree(downloadPath)
    if os.path.exists(imagePath):
        shutil.rmtree(imagePath)

    # data = history.history
    # data["epoch"] = list(range(Epoch))
    # with open("training_log.json", "w") as f:
    #     json.dump({"info": "uploadmodel", "data": data}, f)
    if os.path.split(ossPath)[0] != None:
        storage.upload(
            "{}/training_log.json".format(os.path.split(ossPath)[0]),
            "training_log.json",
        )
    endTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    useTime = time.time() - timeStamp
    trainingInfo["endTime"] = endTime
    trainingInfo["useTime"] = useTime
    with open("traininginfo.json", "w") as f:
        json.dump(trainingInfo, f)
    if os.path.split(ossPath)[0] != None:
        storage.upload(
            "{}/traininginfo.json".format(os.path.split(ossPath)[0]),
            "traininginfo.json",
        )
    # 将 args.outputData1 作为输出发送给下一节点
    return args.outputData1


if __name__ == "__main__":
    Demo()
