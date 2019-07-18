# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String, Json
from suanpan.interfaces import HasArguments
import requests
import json


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
        host = envparam[envparam.index("--stream-host") + 1]
        port = 30007
        templateId = 4407

        urlStatus = "http://{}:{}/app/status".format(host, port)
        dataStatus = {"id": templateId}

        urlRun = "http://{}:{}/app/run".format(host, port)

        if args.inputData1["type"] == "start":
            dataRun = {
                "id": str(templateId),
                "nodeId": "9eaffd90a6eb11e99145cd47c734111d",
                "type": "runStop",
                "setedParams": {
                    "fbfda3609f0411e9a1e9b3f60be454b7": {
                        # 图片
                        "param1": {
                            "value": "studio/{}/{}/{}/predict/{}".format(
                                userId,
                                appId,
                                args.inputData1["programId"],
                                args.inputData1["fileName"],
                            )
                        },
                        # 模型
                        "param2": {
                            "value": "studio/{}/{}/{}/model".format(
                                userId, appId, args.inputData1["programId"]
                            )
                        },
                    },
                    # 预测结果
                    "9eaffd90a6eb11e99145cd47c734111d": {
                        "param1": {
                            "value": "studio/{}/{}/{}/predict/".format(
                                userId, appId, args.inputData1["programId"]
                            )
                        }
                    },
                },
            }

            rStatus = requests.post(url=urlStatus, json=dataStatus)
            print(rStatus)
            print(rStatus.content)
            if json.loads(rStatus.content)["map"]["status"] == "1":
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
            if json.loads(rStatus.content)["map"]["status"] == "3":
                self.send({"status": "success"})
                return None
            elif json.loads(rStatus.content)["map"]["status"] == "1":
                self.send({"status": "running"})
                return None
            elif json.loads(rStatus.content)["map"]["status"] == "4":
                self.send({"status": "fail"})
                return None
            else:
                self.send({"status": "running"})
                return None

        return args.outputData1


if __name__ == "__main__":
    StreamDemo().start()
