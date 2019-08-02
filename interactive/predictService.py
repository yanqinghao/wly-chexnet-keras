# coding=utf-8
from __future__ import absolute_import, print_function

import os
import json
import shutil

from keras.preprocessing import image
import numpy as np
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Json
from suanpan.interfaces import HasArguments
from suanpan.storage import storage
from model import ModelFactory


class StreamDemo(Stream):
    def afterInit(self, *args, **kwargs):  # 初始化模型
        self.model = {"id": None, "model": None, "map": None}

    def loadImage(self, inputImage, imageSize, userId, appId, programId, fileName):
        ossFolder = "studio/{}/{}/{}/predict/{}".format(
            userId, appId, programId, fileName
        )
        print("load image from {}".format(ossFolder))
        storage.downloadFolder(folderName=ossFolder, folderPath=inputImage)
        testPath = inputImage
        testFile = os.listdir(inputImage)
        images = image.load_img(
            os.path.join(testPath, testFile[0]),
            grayscale=False,
            target_size=(imageSize, imageSize),
        )
        x = image.img_to_array(images) / 255.0
        x = np.expand_dims(x, axis=0)
        return x

    def loadModel(self, inputModel, userId, appId, programId, modelName):  # 初始化模型
        self.model["id"] = programId
        ossFolder = "studio/{}/{}/{}/model".format(userId, appId, programId)
        print("load model from {}".format(ossFolder))
        storage.downloadFolder(folderName=ossFolder, folderPath=inputModel)
        output_weights_path = os.path.join(inputModel, "weight.h5")
        with open(os.path.join(inputModel, "label_map.json"), "r") as load_f:
            label_map = json.load(load_f)
        print("load labels : {}".format(label_map))
        self.model["map"] = label_map
        class_names = list(label_map.keys())
        model_factory = ModelFactory()
        self.model["model"] = model_factory.get_model(
            model_name=modelName,
            class_names=class_names,
            use_base_weights=False,
            weights_path=output_weights_path,
        )

    # 定义输入
    @h.input(Json(key="inputData1"))
    # 定义输出
    @h.output(Json(key="outputData1"))
    def call(self, context):
        # 从 Context 中获取相关数据
        args = context.args
        # 查看上一节点发送的 args.inputData1 数据
        print(args.inputData1)
        envparam = HasArguments.getArgListFromEnv()
        userId = envparam[envparam.index("--stream-user-id") + 1]
        appId = envparam[envparam.index("--stream-app-id") + 1]
        programId = args.inputData1["programId"]
        fileName = args.inputData1["fileName"]
        inputImage = "./imagefile"
        inputModel = "./modelfile"
        modelName = "DenseNet121"
        imageSize = 224

        if os.path.exists(inputImage):
            shutil.rmtree(inputImage)

        if os.path.exists(inputModel):
            shutil.rmtree(inputModel)

        if args.inputData1["type"] == "start":
            try:
                if self.model["id"] != programId:
                    self.loadModel(inputModel, userId, appId, programId, modelName)

                x = self.loadImage(
                    inputImage, imageSize, userId, appId, programId, fileName
                )

                result = self.model["model"].predict(x)
                class_names = list(self.model["map"].keys())
                if len(class_names) == 2:
                    result = class_names[1] if result > 0.5 else class_names[0]
                else:
                    result = class_names[np.argsort(result[0])[::-1][0]]
                print("predict result : {}".format(result))

                self.send({"status": "success", "result": result})

                with open("recommendation.json", "w") as f:
                    json.dump({"id": self.model["id"]}, f)
                ossPath = "studio/{}/{}/recommendation.json".format(userId, appId)
                storage.uploadFile(objectName=ossPath, filePath="recommendation.json")

                if os.path.exists(inputImage):
                    shutil.rmtree(inputImage)

                if os.path.exists(inputModel):
                    shutil.rmtree(inputModel)
            except:
                self.send({"status": "failed", "result": None})
        else:
            self.send({"status": "wrong type", "result": None})

        return None


if __name__ == "__main__":
    StreamDemo().start()

