# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder, String
import shutil
import os
from suanpan.storage import storage

# 定义输入
@dc.input(Folder(key="inputData1", required=True))
# 定义输出
@dc.output(Folder(key="outputData1", required=True))
@dc.param(String(key="param1"))
def Demo(context):
    # 从 Context 中获取相关数据
    args = context.args
    # 查看上一节点发送的 args.inputData1 数据
    print(args.inputData1)

    # 自定义代码
    print(args.param1)
    for i in os.listdir(args.inputData1):
        storage.upload(os.path.join(args.param1, i), os.path.join(args.inputData1, i))
        # shutil.copy(os.path.join(args.inputData1, i), os.path.join(args.param1,i))
        # shutil.copy(os.path.join(args.inputData1, i), os.path.join(args.outputData1,i))

    # 将 args.outputData1 作为输出发送给下一节点
    return args.outputData1


if __name__ == "__main__":
    Demo()
