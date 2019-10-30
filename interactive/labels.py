# coding=utf-8
from __future__ import absolute_import, print_function

import json
import random
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Json
from suanpan.interfaces import HasArguments
from suanpan.storage import storage


class StreamDemo(Stream):

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

        ossPath = "studio/{}/{}/{}/detail.json".format(userId, appId, programId)
        storage.download(ossPath, "detail.json")
        with open("detail.json", "r") as f:
            label_detail = json.load(f)

        idx = random.randint(0, len(label_detail["images"]) - 1)
        outputData = args.inputData1
        if "checkType" in label_detail["images"][idx].keys():
            outputData["result"].update({"checkType": label_detail["images"][idx]["checkType"]})
        outputData["result"].update(
            {"imageSharpnessRate": label_detail["images"][idx]["imageSharpnessRate"]}
        )
        outputData["result"].update(
            {
                "deviceCapabilityRate": label_detail["images"][idx][
                    "deviceCapabilityRate"
                ]
            }
        )

        self.send(outputData)

        return None


if __name__ == "__main__":
    StreamDemo().start()

