# Create by Rendawei
# 2019-3-7 11:26

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
    ):

        self.listImagePaths = get_all_files(pathImageDirectory)
        print(len(self.listImagePaths), " images to convert.")
        self.listImageLabels = []
        self.transform = transform
        self.outputPath = outputPath
        # if not (pathImageDirectory == None or pathDatasetFile == None):
        #     # ---- Open file, get image paths and labels
        #     fileDescriptor = open(pathDatasetFile, "r")
        #     # ---- get into the loop
        #     line = True
        #     while line:
        #         line = fileDescriptor.readline()
        #         # --- if not empty
        #         if line:
        #             lineItems = line.split()
        #             imagePath = os.path.join(pathImageDirectory, lineItems[0])
        #             imageLabel = lineItems[1:]
        #             imageLabel = [int(i) for i in imageLabel]
        #             self.listImagePaths.append(imagePath)
        #             self.listImageLabels.append(imageLabel)
        #
        #     fileDescriptor.close()

        self.SERIESES = [
            "W胸部后前位",
            "胸部正位",
            "胸部后前位",
            "Chest",
            "pa",
            "DX-胸部",
            "床边胸片正位",
            "V04_0014",
            "TChestap.",
            "ap",
            "WChestp.a",
            "W033Chestp.a.",
        ]

    # add by rdw 2019-4-2
    def __caculateClassStatic(self):
        nClass = len(self.listImageLabels[0])
        self.listLabelSum = np.zeros([nClass])
        for i in range(len(self.listImageLabels)):
            for j in range(nClass):
                self.listLabelSum[j] = self.listLabelSum[j] + self.listImageLabels[i][j]
        return self.listLabelSum

    # add by rdw 2019-4-2
    def getClassStatic(self):
        if not hasattr(self, "listLabelSum"):
            self.__caculateClassStatic()
        return self.listLabelSum

    # open file
    def transformImageData(self, imageData):
        if self.transform != None:
            imageData = self.transform(imageData)
        return imageData

    def getImageDataFromFile(self, filePath):
        try:
            if filePath.find("png") != -1:
                sucess, imageData = self.__read_png(filePath)
            elif filePath.find("dcm") != -1:
                sucess, imageData = self.__read_dicom(filePath)
            else:
                sucess, imageData = self.__read_dicom(filePath)

            if sucess == False:
                return None
        except:
            # log.error("DatasetGenerator.getImageDataFromFile?error=读取二级制文件出现异常")
            return None
        return imageData

    # 返回PIL格式，１　解析　　２　进行图像转换
    def openFile(self, filePath):
        try:
            if filePath.find("png") != -1:
                sucess, imageData = self.__read_png(filePath)
            elif filePath.find("dcm") != -1:
                sucess, imageData = self.__read_dicom(filePath)
            else:
                sucess, imageData = self.__read_dicom(filePath)

            if sucess == False:
                return None
        except Exception as e:
            # log.error("读取二级制文件出现异常={}".format(e))
            return None
        if self.transform != None:
            imageData = self.transform(imageData)
        return imageData

    # 返回PIL格式，１　解析　　２　进行图像转换
    def openFileReturnFlag(self, filePath):
        try:
            if filePath.find("png") != -1:
                sucess, imageData = self.__read_png(filePath)
            elif filePath.find("dcm") != -1:
                sucess, imageData = self.__read_dicom(filePath)
            else:
                sucess, imageData = self.__read_dicom(filePath)
            if sucess == False:
                return False, imageData
        except Exception as e:
            # log.error("读取二级制文件出现异常={}".format(e))
            return False, imageData
        if self.transform != None:
            imageData = self.transform(imageData)
        return True, imageData

    # 将所有的文件格式修改为png格式
    def convert2Png(self, pngFileBasePath, pngLabelFile, pngNumber=None):
        pngFile = open(pngLabelFile, "a")
        for i in range(len(self.listImagePaths)):
            path = self.listImagePaths[i]
            img_data = self.openFile(path)
            if None == img_data:
                print("解析dicom文件错误, error file continue .... ....")
                continue
            file_name = os.path.basename(path)
            print("{}-{}".format(i, file_name))
            labels = ""
            for j in range(len(self.listImageLabels[i])):
                labels = labels + " {}".format(self.listImageLabels[i][j])

            if pngNumber == None:
                png_file_name = file_name + ".png"
                png_file_path = os.path.join(pngFileBasePath, png_file_name)
                img_data.save(png_file_path, "png")
                label_cell = png_file_name + labels + "\r\n"
                pngFile.write(label_cell)
            else:
                for num in range(pngNumber):
                    png_file_name = "{}-{}.png".format(file_name, num)
                    png_file_path = os.path.join(pngFileBasePath, png_file_name)
                    img_data.save(png_file_path, "png")
                    label_cell = png_file_name + labels + "\r\n"
                    pngFile.write(label_cell)
                    img_data = self.openFile(path)
        pngFile.close()

    def __getitem__(self, index):
        # errorLabel = torch.FloatTensor(np.zeros([100]))
        imagePath = self.listImagePaths[index]
        sucess = False
        # imageLabel = torch.FloatTensor(self.listImageLabels[index])
        try:
            if imagePath.find("png") != -1:
                sucess, imageData = self.__read_png(imagePath)
            elif imagePath.find("dcm") != -1:
                sucess, imageData = self.__read_dicom(imagePath)
            else:
                sucess, imageData = self.__read_dicom(imagePath)
        except:
            pass
            # if sucess == False:
            #     #log.error("返回二级制文件错误,success={}".format(sucess))
            #     return errorLabel, errorLabel
        if type(imageData) == Image.Image:
            print("save image ", index)
            imageData.save("{}/t{}.png".format(self.outputPath, index))
        # except Exception as e:
        #     log.error("读取二级制文件出现异常={}".format(e))
        # return [0],[0]

        # if self.transform != None:
        #     imageData = self.transform(imageData)

        return imageData, []

    # --------------------------------------------------------------------------------
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

        # 当ｓｅｒｉｅｓ　ｄｅｓ
        """
        BeRightProtocol = False
        sdes = dcmFile.SeriesDescription
        for ni in range(len(self.SERIESES)):
            sdes = sdes.replace(' ','')
            if sdes == self.SERIESES[ni]:
                BeRightProtocol = True
                break
        if BeRightProtocol == False:
            #log.error('DatasetGenerator.__read_dicom?error=当前检查部位错误sdes={}'.format(sdes))
            return False, None
        """

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
        # maxDen = len(lut)
        # data[data > maxDen - 1] = maxDen - 1
        # data[data < 0] = 0

        # @1
        # des_data = [lut[data[i][j]] for i in range(data.shape[0]) for j in range(data.shape[1])]
        # des_data = np.array(des_data)
        # des_data = des_data.reshape([data.shape[0], data.shape[1]])
        # @2
        # des_data = cv2.LUT(data,lut)

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


if __name__ == "__main__":
    ds = DatasetGenerator(
        "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\fileparse\\b",
        "C:\\Users\\yanqing.yqh\\code\\wly-chexnet-keras\\fileparse\\b_result",
    )
    for i, d in enumerate(ds):
        print(i + 1, "images done.")
