# coding=utf-8
from __future__ import absolute_import, print_function

import json
import random
import suanpan
from suanpan import g
from suanpan.app import app
from suanpan.storage import storage
from suanpan.stream.arguments import Json


@app.input(Json(key="inputData1"))
@app.output(Json(key="outputData1"))
def labels(self, context):
    args = context.args
    programId = args.inputData1["programId"]
    if args.inputData1["status"] == "success":
        ossPath = "studio/{}/share/{}/uploads/{}/detail.json".format(g.userId, g.appId, programId)
        storage.download(ossPath, "detail.json")
        with open("detail.json", "r") as f:
            label_detail = json.load(f)

        idx = random.randint(0, len(label_detail["images"]) - 1)
        outputData = args.inputData1
        if "checkType" in label_detail["images"][idx].keys():
            outputData["result"].update({"checkType": label_detail["images"][idx]["checkType"]})
        outputData["result"].update(
            {"imageSharpnessRate": label_detail["images"][idx]["imageSharpnessRate"]})
        outputData["result"].update(
            {"deviceCapabilityRate": label_detail["images"][idx]["deviceCapabilityRate"]})

        return outputData
    else:
        return args.inputData1


if __name__ == "__main__":
    suanpan.run(app)
