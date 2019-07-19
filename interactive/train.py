# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String, Json
from suanpan.interfaces import HasArguments
from suanpan.storage import storage
import requests
import json


class StreamDemo(Stream):
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
        port = 30007
        templateId = 4406

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
        if args.inputData1["type"] == "start":
            ossPath = args.inputData1["data"]
            dataRun = {
                "id": "4406",
                "nodeId": "3e2f49509f0511e9a1e9b3f60be454b7",
                "type": "runStart",
                "setedParams": {
                    # 图片路径
                    "f8ca64d09f0411e9a1e9b3f60be454b7": {
                        "param1": {"value": "{}/images".format(ossPath)}
                    },
                    # 模型路径
                    "c415f080a44d11e9a2ebe344cb7c1847": {
                        "param1": {"value": "{}/model".format(ossPath)}
                    },
                },
            }
            osslogFile = "{}/training_log.json".format(ossPath)
            if storage.isFile(objectName=osslogFile):
                storage.removeFile(fileName=osslogFile)
            rStatus = requests.post(url=urlStatus, json=dataStatus)
            print(rStatus)
            print(rStatus.content)
            if json.loads(rStatus.content)["map"]["status"] in ["1", "5", "6", "8", "9"]:
                self.send({"status": "waiting"})
                return None
            else:
                print("start running")
                rRun = requests.post(url=urlRun, json=dataRun)
                print(rRun)
                print(rRun.content)
                self.send({"status": "running"})
                return None
        elif args.inputData1["type"] == "status":
            rStatus = requests.post(url=urlStatus, json=dataStatus)
            print(rStatus)
            print(rStatus.content)
            if json.loads(rStatus.content)["map"]["status"] in status.keys():
                self.send({"status": status[json.loads(rStatus.content)["map"]["status"]]})
                return None
            else:
                self.send({"status": "unknown status"})
                return None

        return args.outputData1


if __name__ == "__main__":
    StreamDemo().start()
