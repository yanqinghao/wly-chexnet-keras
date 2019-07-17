# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String, Json
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
        if args.inputData1["type"] == "start":
            url = "http://221.13.203.139:30007/app/status"
            data = {"id": 4407}
            r = requests.post(url=url, json=data)
            print(r)
            print(r.content)
            if json.loads(r.content)["map"]["status"] == "1":

                self.send({"status": "waiting"})
                return None
            else:
                print("start running")
                # print(args.inputData2["data"])
                url = "http://221.13.203.139:30007/app/run"
                headers = {"Content-Type": "application/json"}
                data = {
                    "id": "4407",
                    "nodeId": "9eaffd90a6eb11e99145cd47c734111d",
                    "type": "runStop",
                    "setedParams": {
                        # 图片
                        "fbfda3609f0411e9a1e9b3f60be454b7": {
                            "param1": {
                                "value": "studio/100090/4438/"
                                + str(args.inputData1["programId"])
                                + "/predict/"
                                + args.inputData1["fileName"]
                                # + ".png"
                            },
                            # 模型
                            "param2": {
                                "value": "studio/100090/4438/"
                                + str(args.inputData1["programId"])
                                + "/model"
                            }
                        },
                        "9eaffd90a6eb11e99145cd47c734111d": {
                            "param1": {
                                "value": "studio/100090/4438/"
                                + str(args.inputData1["programId"])
                                + "/predict/"
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
            data = {"id": 4407}
            r = requests.post(url=url, json=data)
            print(r)
            print(r.content)
            if json.loads(r.content)["map"]["status"] == "3":
                self.send({"status": "success"})
                return None
            elif json.loads(r.content)["map"]["status"] == "1":
                self.send({"status": "running"})
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
