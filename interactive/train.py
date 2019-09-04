# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import Int, Json, String
from suanpan.interfaces import HasArguments
from suanpan.storage import storage
import requests
import json


class StreamDemo(Stream):
    ARGUMENTS = [
        Int(key="param1", default=30007),
        Int(key="param2", default=4406),
        String(key="param3", default="f8ca64d09f0411e9a1e9b3f60be454b7"),
        String(key="param4", default="c415f080a44d11e9a2ebe344cb7c1847"),
    ]
    # 定义输入
    @h.input(Json(key="inputData1", required=True))
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
        host = envparam[envparam.index("--stream-host") + 1]
        port = args.param1
        templateId = args.param2

        urlStatus = "http://{}:{}/app/status".format(host, port)
        dataStatus = {"id": templateId}

        urlRun = "http://{}:{}/app/run".format(host, port)

        status = {
            "0": "none",
            "1": "running",
            "2": "stopped",
            "3": "success",
            "4": "failed",
            "5": "starting",
            "6": "stopping",
            "7": "dead",
            "8": "cron",
            "9": "waiting",
        }
        # app 处于 cron 状态，就不会被运行
        if "trainingId" in list(args.inputData1.keys()):
            trainingId = args.inputData1["trainingId"]
        else:
            trainingId = None
        if args.inputData1["type"] == "start":
            ossPath = args.inputData1["data"]
            try:
                trainTest = str(args.inputData1["xunLian"] / 100)
            except:
                trainTest = "0.8"
            dataRun = {
                "id": str(args.param2),
                "nodeId": args.param3,
                "type": "runStart",
                "setedParams": {
                    # 图片路径
                    args.param3: {
                        "param1": {"value": "{}/images".format(ossPath)},
                        "param2": {"value": trainTest},
                    },
                    # 模型路径
                    args.param4: {"param1": {"value": "{}/model".format(ossPath)}},
                },
            }
            osslogFile = "{}/training_log.json".format(ossPath)
            if storage.isFile(osslogFile):
                storage.remove(osslogFile)
            rStatus = requests.post(url=urlStatus, json=dataStatus)
            print(rStatus)
            print(rStatus.content)
            if json.loads(rStatus.content)["map"]:
                if json.loads(rStatus.content)["map"]["status"] in [
                    "1",
                    "5",
                    "6",
                    "8",
                    "9",
                ]:
                    self.send({"status": "waiting", "trainingId": trainingId})
                    return None
                else:
                    print("start running")
                    rRun = requests.post(url=urlRun, json=dataRun)
                    print(rRun)
                    print(rRun.content)
                    self.send({"status": "running", "trainingId": trainingId})
                    return None
            else:
                print("First time start running")
                rRun = requests.post(url=urlRun, json=dataRun)
                print(rRun)
                print(rRun.content)
                self.send({"status": "running", "trainingId": trainingId})
                return None
        elif args.inputData1["type"] == "status":
            rStatus = requests.post(url=urlStatus, json=dataStatus)
            print(rStatus)
            print(rStatus.content)
            if json.loads(rStatus.content)["map"]:
                if json.loads(rStatus.content)["map"]["status"] in status.keys():
                    self.send(
                        {
                            "status": status[
                                json.loads(rStatus.content)["map"]["status"]
                            ],
                            "trainingId": trainingId,
                        }
                    )
                    return None
                else:
                    self.send({"status": "unknown status", "trainingId": trainingId})
                    return None
            else:
                self.send({"status": "unknown status", "trainingId": trainingId})
                return None
        else:
            self.send({"status": "unknown type", "trainingId": trainingId})
            return None

        return args.outputData1


if __name__ == "__main__":
    StreamDemo().start()
