import numpy as np
from PIL import Image
from pathlib import Path
from skimage import measure
from labelme import utils
import cv2

sourcePath_dem = Path.cwd() / 'original/DEMImages_jpg'
sourcePath_demTif = Path.cwd() / 'original/DEMImages_tif'

resPath_contourLine = Path.cwd() / 'optimize/mountain_contourLine'
resPath_mountainTemp = Path.cwd() / 'optimize/mountain_temp'
resPath_mountain = Path.cwd() / "optimize/mountain"
resPath_altitute  = Path.cwd() / 'optimize/altitute'

def get_contourLine():
    """
    根据DEM图获得山脚线
    """
    for file in sourcePath_dem.iterdir():
        print(file.name)
        image = np.array(Image.open(file))
        res = np.zeros_like(image, dtype=np.uint8)
        image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))
        # Construct some test data
        x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
        r = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))

        # Find contours at a constant value of 0.8
        contours = measure.find_contours(image_norm, 0.3, fully_connected='high', positive_orientation='high')
        contours = sorted(contours, key=lambda x:x.shape, reverse=True)[:5]

        for contour in contours:
            contour = contour.astype(np.int32)
            for point in contour:
                res[point[0], point[1]] = 255
        res_image = Image.fromarray(res)
        res_image.save(resPath_contourLine / file.name)

def get_plain():
    """
    优化山脚线得到山体、平原区域
    """
    for file in resPath_contourLine.iterdir():
        print(file.name)
        image = np.array(Image.open(file))
        kernel = np.ones((5, 5), np.uint8)
        # 中值滤波平滑边缘
        image = cv2.medianBlur(image, 17)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
        image = Image.fromarray(image)
        image.save(resPath_mountainTemp / file.name, quantile=95)

def split_altitute():
    """
    根据高度值对dem进行划分
    """
    for file in sourcePath_demTif.iterdir():
        # print(file.name)
        image = np.array(Image.open(file))
        print("{}: ({}, {})".format(file.stem, np.min(image).astype(int), np.max(image).astype(int)))
        res = np.zeros_like(image, dtype=np.uint8)
        res[image > 500] = 255
        res = Image.fromarray(res)
        res.save(resPath_altitute / (file.stem + '.jpg'), quantile=95)

def convert():
    """
    转换文件为统一格式
    """
    for file in resPath_mountainTemp.iterdir():
        print(file.name)
        image = np.array(Image.open(file).convert("1"))
        res = np.zeros_like(image, dtype=np.uint8)
        res[image>0] = 1
        utils.lblsave(resPath_mountain / (file.stem + '.png'), res)

if __name__ == "__main__":
    convert()
