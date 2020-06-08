# coding=utf-8
from __future__ import absolute_import, print_function

import suanpan
from suanpan.app import app
from suanpan.log import logger
from suanpan.storage import storage
from suanpan.stream.arguments import Int, Json, String
from suanpan.api import post


@app.input(Json(key="inputData1", required=True))
@app.param(Int(key="param1"))
@app.param(Int(key="param2", default=4406))
@app.param(String(key="param3", default="f8ca64d09f0411e9a1e9b3f60be454b7"))
@app.param(String(key="param4", default="c415f080a44d11e9a2ebe344cb7c1847"))
@app.param(String(key="param5", default="http"))
@app.output(Json(key="outputData1"))
def trainNew(context):
    args = context.args
    templateId = args.param2

    urlStatus = "/app/status"
    dataStatus = {"id": templateId}
    urlRun = "/app/run"

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
                    "param1": {
                        "value": "{}/images".format(ossPath)
                    },
                    "param2": {
                        "value": trainTest
                    },
                },
                # 模型路径
                args.param4: {
                    "param1": {
                        "value": "{}/model".format(ossPath)
                    }
                },
            },
        }
        osslogFile = "{}/training_log.json".format(ossPath)
        if storage.isFile(osslogFile):
            storage.remove(osslogFile)
        rStatus = post(urlStatus, json=dataStatus)
        logger.info(rStatus)
        if rStatus["map"]:
            if rStatus["map"]["status"] in [
                    "1",
                    "5",
                    "6",
                    "8",
                    "9",
            ]:
                app.send({"status": "waiting", "trainingId": trainingId})
            else:
                logger.info("start running")
                rRun = post(urlRun, json=dataRun)
                logger.info(rRun)
                app.send({"status": "running", "trainingId": trainingId})
        else:
            logger.info("First time start running")
            rRun = post(urlRun, json=dataRun)
            logger.info(rRun)
            app.send({"status": "running", "trainingId": trainingId})
    elif args.inputData1["type"] == "status":
        rStatus = post(urlStatus, json=dataStatus)
        logger.info(rStatus)
        if rStatus["map"]:
            if rStatus["map"]["status"] in status.keys():
                app.send({
                    "status": status[rStatus["map"]["status"]],
                    "trainingId": trainingId,
                })
            else:
                app.send({"status": "unknown status", "trainingId": trainingId})
        else:
            app.send({"status": "unknown status", "trainingId": trainingId})
    else:
        app.send({"status": "unknown type", "trainingId": trainingId})

    return args.outputData1


if __name__ == "__main__":
    suanpan.run(app)
