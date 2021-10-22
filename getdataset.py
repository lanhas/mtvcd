from os import path
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.lib.function_base import quantile
from numpy.lib.type_check import imag


TARGET_SIZE = (2448, 2448)      # 目标大小（宽，高）
original_fixPath = Path('original')     # 从91卫图下载的原始文件
original_remotePath = Path('remoteData')    # 分离后得到的原始遥感数据
original_demPath = Path('demData')      # 分离后得到的原始高程数据
target_remotePath = Path('JPEGImages')  # 目标遥感数据，2448*2448，jpg文件
target_demPath_tif = Path('DEMImages_tif')      # 目标高程数据，2448*2448，tif文件，原始高程
target_demPath_jpg = Path('DEMImages_jpg')      # 目标高程数据，2448*2448，jpg文件，压缩高程

def separate_file():
    """
    分离文件，将下载后的混合文件分离
    """
    for _, fileName in enumerate(original_fixPath.iterdir()):
        for _, v in enumerate(fileName.iterdir()):
            if v.is_dir():
                for _, fileName in enumerate(v.iterdir()):
                    if fileName.name.split('_')[1] == '影像.tif':

                        newPath = original_remotePath / fileName.name.split('_')[0] + '_sat.tif'
                        fileName.replace(newPath)
                    elif fileName.name.split('_')[1] == '高程.tif':
                        newPath = original_demPath / fileName.name.split('_')[0] + '_dem.tif'
                        fileName.replace(newPath)

def getResult():
    """
    将dem数据与遥感数据缩放到并裁剪到固定大小（同box大小相同)
    数据类型:
        dem：.tif格式，大小152*152, 168*151等, 精度8.5米/像素
        遥感数据：.tif格式，大小为2560*2560， 2304*2304, 2304*2560等， 精度0.53米/像素
    """
    target_width = TARGET_SIZE[0]
    target_height = TARGET_SIZE[1]
    box = (0, 0, target_height, target_width)    # 模板
    dem_list = []   # dem文件名（不含后缀）
    jpeg_list = []  #遥感数据文件名（不含后缀）
    # 检查dem与遥感数据是否匹配
    for _, v in enumerate(original_demPath.iterdir()):
        dem_list.append(v.name.split('_')[0])
    for _, v in enumerate(original_remotePath.iterdir()):
        jpeg_list.append(v.name.split('_')[0])
    if not dem_list==jpeg_list:
        raise ValueError('dem与遥感数据不匹配，请检查！')
    # 改变数据大小
    for _, fileName in enumerate(dem_list):
        jpeg_file = Image.open(original_remotePath / (fileName + '_sat.tif'))
        dem_file = Image.open(original_demPath / (fileName + '_dem.tif'))
        img_width = jpeg_file.width
        img_height = jpeg_file.height 
        dem_file = dem_file.resize((img_width, img_height))
        # result = dem2bmp(dem_file)
        # result.save('DEMImages/' + fileName + '_dem.jpg', quality=95)

        # 宽高均大于目标大小，裁剪图片
        if img_width > target_width and img_height > 2448:
            jpeg_file = jpeg_file.crop(box)
            dem_file = dem_file.crop(box)
        # 宽高均小于目标大小，resize图片
        elif img_width < target_width and img_height < target_height:
            jpeg_file = jpeg_file.resize((target_width, target_height))
            dem_file = dem_file.resize((target_width, target_height))
        # 一边大于目标大小，一边小于目标大小，将大的一头先裁剪同小的一样，再resize   
        else:
            if img_width > img_height:
                dem_file = dem_file.crop((0, 0, img_height, img_height))
                dem_file = dem_file.resize((target_width, target_height))
                jpeg_file = jpeg_file.crop((0, 0, img_height, img_height))
                jpeg_file = jpeg_file.resize((target_width, target_height))
            else:
                dem_file = dem_file.crop((0, 0, img_width, img_width))
                dem_file = dem_file.resize((target_width, target_height))
                jpeg_file = jpeg_file.crop((0, 0, img_width, img_width))
                jpeg_file = jpeg_file.resize((target_width, target_height))
        remote_jpg = jpeg_file
        dem_tif = dem_file
        dem_jpg = tif2bmp(dem_file)
        # 保存文件
        remote_jpg.save(target_remotePath / (fileName + '_sat.jpg'), quality=95)
        dem_tif.save(target_demPath_tif / (fileName + '_dem.tif'), quality=95)
        dem_jpg.save(target_demPath_jpg / (fileName + '_dem.jpg'), quality=95)

def result2label(source, target):
    """
    将图像转为训练用的label格式，1~n

    Parameters
    ----------
    source: pathlib Path
        源文件夹
    target: pathlib Path
        目标文件夹
    """
    for _, val in enumerate(source.iterdir()):
        print(val.name)
        image = Image.open(val)
        image = np.array(image, dtype=np.uint8)
        image[image==255] = 1
        result = Image.fromarray(image)
        fileName = target / val.name
        result.save(fileName, quantile=95)
 

def tif2bmp(img: Image)->Image:
    """
    将高程tif文件转为bmp文件，压缩方法：
        255 * (currentNum - minNum)/(maxNum - minNum)
    """
    dem_array = np.array(img)
    dem_min = dem_array.min()
    dem_max = dem_array.max()
    dem_array = 255 * np.divide(dem_array - dem_min, dem_max - dem_min)
    dem_array = dem_array.astype(np.uint8)
    result = Image.fromarray(dem_array)
    return result

if __name__ == "__main__":
    # getResult()
    source = Path('mountain')
    target = Path('SegmentationClass')
    result2label(source, target)
