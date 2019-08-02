# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.docker import DockerComponent as dc
from suanpan.docker.arguments import Folder
from suanpan.stream import Handler as h
from suanpan.stream import Stream
from suanpan.stream.arguments import String, Json
from suanpan.interfaces import HasArguments
import os
import pdb
import sys
import glob
import time
import shutil
import logging
import pydicom as dicom
import numpy as np
import cv2
import SimpleITK as sitk
from multiprocessing.dummy import Pool
import torch
from torch.utils.data import Dataset
from PIL import Image
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.image as mat_img
from torch.utils.data import DataLoader
from suanpan.storage import storage
import base64
import json


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_


class DatasetGenerator(Dataset):
    """
    pathImageDirectory:文件路径目录
    pathDatasetFile：数据集文件
    transform：转换
    """

    def __init__(
        self,
        pathImageDirectory=None,
        outputPath="./",
        pathDatasetFile=None,
        transform=None,
        userId=None,
        appId=None,
        programId=None,
    ):
        self.pathImageDirectory = pathImageDirectory
        if os.path.isdir(self.pathImageDirectory):
            print("************prepare dir list**************")
            self.listImagePaths = get_all_files(self.pathImageDirectory)
        else:
            print("**************prepare file****************")
            self.listImagePaths = [self.pathImageDirectory]
        print(len(self.listImagePaths), " images to convert.")
        self.listImageLabels = []
        self.transform = transform
        self.outputPath = outputPath
        self.userId = userId
        self.appId = appId
        self.programId = programId

    def __getitem__(self, index):
        print("********start********" + str(index))
        imagePath = self.listImagePaths[index]
        filePng = "studio/{}/{}/{}/images/{}.png".format(
            self.userId, self.appId, self.programId, imagePath[7:]
        )
        filecheck = storage.isFile(objectName=filePng)
        if not filecheck:
            try:
                if imagePath.find("png") != -1:
                    sucess, imageData = self.__read_png(imagePath)
                elif imagePath.find("dcm") != -1:
                    sucess, imageData = self.__read_dicom(imagePath)
                else:
                    print("no name")
                    sucess, imageData = self.__read_dicom(imagePath)
            except:
                print("except")
                pass
            if type(imageData) == Image.Image:
                print("save image ", index)
                if os.path.isdir(self.pathImageDirectory):
                    print(imagePath[6:])
                    filepath = imagePath[6:]
                else:
                    filepath = os.path.split(self.pathImageDirectory)[1]
                print(self.outputPath + filepath + ".png")
                pngpath = self.outputPath + filepath + ".png"
                pngdir = os.path.split(pngpath)
                if not os.path.exists(pngdir[0]):
                    os.makedirs(pngdir[0])
                print(pngdir[0])
                imageData.save(pngpath)
                filePng = "studio/{}/{}/{}/images/{}.png".format(
                    self.userId, self.appId, self.programId, imagePath[7:]
                )
                storage.uploadFile(objectName=filePng, filePath=pngpath)

        return None

    def __len__(self):
        return len(self.listImagePaths)

    @jit
    def __autoWl(self, imageData):
        _5h = int(imageData.shape[0] * 0.5)
        _5w = int(imageData.shape[1] * 0.5)
        begin_h = int((imageData.shape[0] - _5h) * 0.5)
        begin_w = int((imageData.shape[1] - _5w) * 0.5)
        end_h = begin_h + _5h
        end_w = begin_w + _5w

        max = np.zeros([8])
        min = np.array(65536)
        min = np.repeat(min, 8, 0)

        hOffset = _5h * 0.125
        for h in range(8):
            hb = int(begin_h + h * hOffset)
            eb = int(hb + hOffset)
            for num_h in range(hb, eb):
                for num_w in range(begin_w, end_w):
                    if max[h] < imageData[num_h][num_w]:
                        max[h] = imageData[num_h][num_w]
                    if min[h] > imageData[num_h][num_w]:
                        min[h] = imageData[num_h][num_w]
        max.sort(axis=0)
        self.windowCenter = int((max[6] - min[1]) * 0.5 + min[1] + 1)
        self.windowWidth = int(max[6] - min[1] + 1)
        # log.info("auto window level w:{};l:{}".format(self.windowWidth,self.windowCenter))

    # read png
    def __read_png(self, imagePath):
        imageData = Image.open(imagePath).convert("RGB")
        return True, imageData

    def _sitk_read(self, img_path):
        ds = sitk.ReadImage(img_path)
        ImagePixData = sitk.GetArrayFromImage(ds)  # sitk read HU value
        return ImagePixData

    def __read_dicom(self, imagePath):
        start_time = time.time()
        """read dicoms in one folders, return one seriesuid slices image"""
        try:
            dcmFile = dicom.read_file(imagePath)
        except Exception as e:
            print("file read fail")
            # log.error('__read_dicom?读取文件错误')
            return False, "读取文件格式错误"

        # 设备类型
        BeRightModality = False
        modality = dcmFile.Modality
        if hasattr(dcmFile, "Modality"):
            if modality == "DX" or "DR" == modality or "CR" == modality:
                BeRightModality = True
            # log.error("DatasetGenerator.__read_dicom?error=当前图像Modality=空")
        if BeRightModality == False:
            # log.error('DatasetGenerator.__read_dicom?error=当前设备类型错误{}'.format(modality))
            return False, "DX DR CR类型的图像Modality={}".format(modality)

        needWL = False
        if hasattr(dcmFile, "BitsAllocated"):
            self.allocByte = int((dcmFile.BitsAllocated + 7) // 8)
            if self.allocByte == 2:
                needWL = True
        self.bitsAllocated = dcmFile.BitsAllocated

        # 最大最小值 , 如果缺少窗宽和窗位　直接赋值
        minVal = 0
        maxVal = 2 ** int(dcmFile.BitsStored)
        self.windowCenter = int(maxVal * 0.5)
        self.windowWidth = maxVal - 1
        self.bitsStored = dcmFile.BitsStored

        hasWCAttribute = True
        if hasattr(dcmFile, "WindowCenter"):
            window_center = dcmFile.WindowCenter
            if isinstance(window_center, dicom.multival.MultiValue):
                self.windowCenter = int(window_center[0])
            elif isinstance(window_center, dicom.valuerep.DSfloat):
                self.windowCenter = int(window_center)
            else:
                hasWCAttribute = False
                # log.info('__read_dicom？ window center 格式不正确')
        else:
            hasWCAttribute = False
            # log.info('__read_dicom?没有窗位信息')

        if hasattr(dcmFile, "WindowWidth"):
            window_width = dcmFile.WindowWidth
            if isinstance(window_width, dicom.multival.MultiValue):
                self.windowWidth = int(window_width[0])
            elif isinstance(window_width, dicom.valuerep.DSfloat):
                self.windowWidth = int(window_width)
            else:
                hasWCAttribute = False
                # log.info('__read_dicom？ window width 格式不正确')
        else:
            hasWCAttribute = False
            # log.info('__read_dicom?没有窗宽信息')

        if not hasattr(dcmFile, "PixelRepresentation"):
            # log.error('__read_dicom?没有PixelRepresentation信息')
            return False, "没有PixelRepresentation信息"

        PhotometricInterpretation = "MONOCHROME1"
        if hasattr(dcmFile, "PhotometricInterpretation"):
            PhotometricInterpretation = dcmFile.PhotometricInterpretation

        try:
            imageData = dcmFile.pixel_array
        except:
            imageData = self._sitk_read(imagePath)  # compress lossless format
            imageData = imageData.reshape(imageData.shape[1], imageData.shape[2])

        # 如果没有窗宽窗位
        if hasWCAttribute == False and needWL == True:
            self.__autoWl(imageData)

        if needWL == True:
            if PhotometricInterpretation == "MONOCHROME1":
                imageData = self.__process_data(imageData, True)
            else:
                imageData = self.__process_data(imageData, False)

        imageData = np.expand_dims(imageData, -1)
        imageData = np.repeat(imageData, 3, 2)

        # 将darray类型转化为PILImage类型
        imageData = Image.fromarray(imageData)
        end_time = time.time()
        # print("time cost is:",end_time - start_time)
        return True, imageData

    @jit
    def __generate_lut(self, invert=True):
        w_left = int(self.windowCenter - self.windowWidth * 0.5)
        w_right = int(self.windowCenter + self.windowWidth * 0.5)
        windowWidth = w_right - w_left
        # len = 2 ** self.bitsStored
        len = 2 ** self.bitsAllocated
        self.a_min = 0
        self.a_max = len - 1
        self.lut = np.zeros((len), np.uint8)
        if invert == False:
            for i in range(len):
                if i < w_left:
                    self.lut[i] = 0
                elif i > w_right:
                    self.lut[i] = 255
                else:
                    self.lut[i] = int((i - w_left) * 255 / windowWidth)
        else:
            for i in range(len):
                if i < w_left:
                    self.lut[i] = 255
                elif i > w_right:
                    self.lut[i] = 0
                else:
                    self.lut[i] = int(255 - ((i - w_left) * 255 / windowWidth))
        return self.lut

    @jit
    def __process_data(self, data, invert):
        # normalization
        des_data_old = np.zeros(data.shape[1] * data.shape[0], np.uint8).reshape(
            [data.shape[0], data.shape[1]]
        )

        lut = self.__generate_lut(invert)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                des_data_old[i][j] = lut[data[i][j]]

        return des_data_old

    def _read_itk(self, filename):
        """ return img, origin, spacing, isflip, all in z, y, x order!
        """
        with open(filename) as f:
            contents = f.readlines()
            line = [k for k in contents if k.startswith("TransformMatrix")][0]
            transformM = np.array(line.split(" = ")[1].split(" ")).astype("float")
            transformM = np.round(transformM)
            if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
                isflip = True
            else:
                isflip = False

        itkimage = sitk.ReadImage(filename)
        numpyImage = sitk.GetArrayFromImage(itkimage)

        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
        seriesuid = os.path.basename(filename)
        seriesuid = os.path.splitext(seriesuid)[0]

        # return numpyImage, numpyOrigin, numpySpacing, isflip, seriesuid

        img_dict = {}
        img_dict["image"] = numpyImage
        # img_dict['slices'] = slices
        img_dict["origin"] = numpyOrigin
        img_dict["spacing"] = numpySpacing
        img_dict["isflip"] = isflip
        img_dict["seriesuid"] = seriesuid

        return img_dict


class FolderException(Exception):
    pass


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
        programId = args.inputData1["id"]
        filePathDcom = "studio/{}/{}/{}/dcom".format(userId, appId, programId)
        osslogFile = "studio/{}/{}/{}/parsinglog.json".format(userId, appId, programId)
        # 自定义代码
        localDcom = "./dcom"
        localPng = "./images"
        logFile = "./parsinglog.json"

        if storage.isFile(objectName=osslogFile):
            storage.removeFile(fileName=osslogFile)

        with open(logFile, "w") as f:
            json.dump({"status": "running"}, f)
        storage.uploadFile(objectName=osslogFile, filePath=logFile)

        if os.path.exists(localDcom):
            shutil.rmtree(localDcom)
        if os.path.exists(localPng):
            shutil.rmtree(localPng)
        try:
            if storage.isFolder(folderName=filePathDcom):
                storage.downloadFolder(folderName=filePathDcom, folderPath=localDcom)
            else:
                raise FolderException("No Dcom Folder")

            ds = DatasetGenerator(
                pathImageDirectory=localDcom,
                outputPath=localPng,
                userId=userId,
                appId=appId,
                programId=programId,
            )

            fileLen = len(ds.listImagePaths)
            with open(logFile, "w") as f:
                json.dump({"status": "running", "now": 0, "fileNum": fileLen}, f)

            storage.uploadFile(objectName=osslogFile, filePath=logFile)
            for i, d in enumerate(ds):
                with open(logFile, "w") as f:
                    json.dump(
                        {"status": "running", "now": i + 1, "fileNum": fileLen}, f
                    )

                storage.uploadFile(objectName=osslogFile, filePath=logFile)
                print(i + 1, "images done.")

            if os.path.exists(localPng):
                shutil.rmtree(localPng)
            if os.path.exists(localDcom):
                shutil.rmtree(localDcom)

            with open(logFile, "w") as f:
                json.dump({"status": "success", "now": i + 1, "fileNum": fileLen}, f)
            storage.uploadFile(objectName=osslogFile, filePath=logFile)
            self.send(args.inputData1)
        except FolderException as fe:
            print("Exception", fe)
            with open(logFile, "w") as f:
                json.dump({"status": "failed", "message": "EmptyFolder"}, f)
            storage.uploadFile(objectName=osslogFile, filePath=logFile)
        except Exception as e:
            print("Exception", e)
            with open(logFile, "w") as f:
                json.dump({"status": "failed", "message": "Exception"}, f)
            storage.uploadFile(objectName=osslogFile, filePath=logFile)

        return None


if __name__ == "__main__":
    StreamDemo().start()
