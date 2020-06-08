# coding=utf-8
from __future__ import absolute_import, print_function

import os
import suanpan
from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from suanpan.storage import storage


@app.input(Folder(key="inputData1"))
@app.output(String(key="outputData1"))
@app.param(String(key="param1"))
def upload(context):
    args = context.args

    print(args.param1)
    for i in os.listdir(args.inputData1):
        storage.upload(os.path.join(args.param1, i), os.path.join(args.inputData1, i))

    return "success"


if __name__ == "__main__":
    suanpan.run(app)
