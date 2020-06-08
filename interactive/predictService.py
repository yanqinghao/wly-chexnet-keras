# coding=utf-8
from __future__ import absolute_import, print_function

import os
import json
import shutil
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import suanpan
from suanpan import g
from suanpan.app import app
from suanpan.log import logger
from suanpan.storage import storage
from suanpan.stream.arguments import Json


def loadImage(inputImage, imageSize, userId, appId, programId, fileName):
    ossFolder = "studio/{}/share/{}/uploads/{}/predict/{}.png".format(
        userId, appId, programId, fileName)
    logger.info("load image from {}".format(ossFolder))
    storage.download(ossFolder, inputImage)
    testPath = inputImage
    testFile = testPath
    images = image.load_img(testFile, grayscale=False, target_size=(imageSize, imageSize))
    x = image.img_to_array(images) / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def loadModel(inputModel, userId, appId, programId, modelName):  # 初始化模型
    g.model["id"] = programId
    ossFolder = "studio/{}/share/{}/uploads/{}/model".format(userId, appId, programId)
    logger.info("load model from {}".format(ossFolder))
    storage.download(ossFolder, inputModel)
    output_weights_path = os.path.join(inputModel, "weight.h5")
    with open(os.path.join(inputModel, "label_map.json"), "r") as load_f:
        label_map = json.load(load_f)
    logger.info("load labels : {}".format(label_map))
    g.model["map"] = label_map
    g.model["model"] = load_model(output_weights_path)


@app.afterInit
def afterInit(context):
    g.model = {"id": None, "model": None, "map": None}


@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def predictService(context):
    args = context.args

    programId = args.inputData1["programId"]
    fileName = args.inputData1["fileName"]
    inputImage = "/tmp/imagefile/predict.png"
    inputModel = "/tmp/modelfile"
    modelName = "DenseNet121"
    imageSize = 224

    if os.path.exists(inputImage):
        os.remove(inputImage)

    if os.path.exists(inputModel):
        shutil.rmtree(inputModel)

    if args.inputData1["type"] == "start":
        try:
            if g.model["id"] != programId:
                loadModel(inputModel, g.userId, g.appId, programId, modelName)

            x = loadImage(inputImage, imageSize, g.userId, g.appId, programId, fileName)

            result = g.model["model"].predict(x)
            class_names = list(g.model["map"].keys())
            if len(class_names) == 2:
                result = class_names[1] if result > 0.5 else class_names[0]
            else:
                result = class_names[np.argsort(result[0])[::-1][0]]
            logger.info("predict result : {}".format(result))

            app.send({
                "status": "success",
                "result": {
                    "rate": result
                },
                "programId": programId,
            })

            with open("recommendation.json", "w") as f:
                json.dump({"id": g.model["id"]}, f)
            ossPath = "studio/{}/share/{}/uploads/recommendation.json".format(g.userId, g.appId)
            storage.upload(ossPath, "recommendation.json")

            if os.path.exists(inputImage):
                os.remove(inputImage)

            if os.path.exists(inputModel):
                shutil.rmtree(inputModel)
        except:
            app.send({"status": "failed", "result": None, "programId": programId})
    else:
        app.send({"status": "wrong type", "result": None, "programId": programId})


if __name__ == "__main__":
    suanpan.run(app)
