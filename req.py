# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String,Json
import requests

class StreamDemo(Stream):
    # 定义输入
    @h.input(Json(key="inputData1", required=True))
    # 定义输出
    @h.output(String(key="outputData1"))
    def call(self, context):
        # 从 Context 中获取相关数据
        args = context.args
        # 查看上一节点发送的 args.inputData1 数据
        print(args.inputData1)
        if args.inputData1["type"] == "start":
            print(args.inputData1["datain"])
            print(args.inputData1["dataout"])
            url = "http://spnext.xuelangyun.com/web/app/4389/app/run"
            headers = {"Content-Type":"application/json"}
            data = {"id":4389,
            "nodeId": "3e2f49509f0511e9a1e9b3f60be454b7",
            "runType": "runStart",
            "setedParams": {
                "3e2f49509f0511e9a1e9b3f60be454b7": {
                    "outPutData1": {
                    "value": "studio/shanglu/1123/project/ssss"
                    },
                "c415f080a44d11e9a2ebe344cb7c1847":{
                    "param1":{
                        "value":"studio/shanglu/1123/project/ssss"
                    }
                }

                }
            }}
            r = requests.post(url=url, data=data, headers=headers)
            # print(r)
            print("ssss")

        # 自定义代码

        # 将 args.inputData1 作为输出发送给下一节点
        return args.inputData1


if __name__ == "__main__":
    StreamDemo().start()
