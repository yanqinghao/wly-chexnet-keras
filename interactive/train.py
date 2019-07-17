# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String, Json
import requests
import json


class StreamDemo(Stream):
    # 定义输入
    @h.input(Json(key="inputData1", required=True))
    # @h.input(Json(key="inputData2", required=True))
    # 定义输出
    @h.output(Json(key="outputData1"))
    def call(self, context):
        # 从 Context 中获取相关数据
        args = context.args
        # 查看上一节点发送的 args.inputData1 数据
        print(args.inputData1)
        # start status
        if args.inputData1["type"] == "start":
            url = "http://221.13.203.139:30007/app/status"
            data = {"id": 4406}
            r = requests.post(url=url, json=data)
            print(r)
            print(r.content)
            if json.loads(r.content)["map"]["status"] == "1":
                self.send({"status": "waiting"})
                return None
            else:
                print("start running")
                print(args.inputData1["data"])
                url = "http://221.13.203.139:30007/app/run"
                headers = {"Content-Type": "application/json"}
                data = {
                    "id": "4406",
                    "nodeId": "3e2f49509f0511e9a1e9b3f60be454b7",
                    "type": "runStart",
                    "setedParams": {
                        # 图片路径
                        "f8ca64d09f0411e9a1e9b3f60be454b7": {
                            "param1": {
                                "value": args.inputData1["data"]
                                + "/images"
                            }
                        },
                        # 模型路径
                        "c415f080a44d11e9a2ebe344cb7c1847": {
                            "param1": {
                                "value": args.inputData1["data"]
                                + "/model"
                            }
                        },
                    },
                }
                r = requests.post(url=url, json=data)
                print(r)
                print(r.content)
                self.send({"status": "running"})
                return None
        elif args.inputData1["type"] == "status":
            url = "http://221.13.203.139:30007/app/status"
            data = {"id": 4406}
            r = requests.post(url=url, json=data)
            print(r)
            print(r.content)
            if json.loads(r.content)["map"]["status"] == "1":
                self.send({"status": "running"})
                return None
            elif json.loads(r.content)["map"]["status"] == "3":
                self.send({"status": "success"})
                return None
            elif json.loads(r.content)["map"]["status"] == "4":
                self.send({"status": "fail"})
                return None
            else:
                self.send({"status": "running"})
                return None

        # 自定义代码

        # 将 args.inputData1 作为输出发送给下一节点
        return args.outputData1


if __name__ == "__main__":
    StreamDemo().start()
