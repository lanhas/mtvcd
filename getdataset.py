from os import path
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.lib.function_base import quantile
from numpy.lib.type_check import imag


targetSize_image = (2448, 2448)                 # 遥感图像目标大小（宽，高）
targetSize_dem = (152, 152)                     # 高数数据目标大小 (宽，高)

sourcePath_fixed = Path.cwd() / 'original'      # 从91卫图下载的原始文件
sourcePath_image = Path.cwd() / 'original/remoteData'    # 分离后得到的原始遥感数据
sourcePath_dem = Path.cwd() / 'original/demData'         # 分离后得到的原始高程数据

targetPath_image = Path('original/JPEGImages')           # 目标遥感数据，2448*2448，jpg文件
targetPath_dem = Path('original/ElevationData')          # 目标高程数据，152*152，tif文件

def separate_file():
    """
    分离文件，将下载后的混合文件分离
    """
    for _, fileName in enumerate(sourcePath_fixed.iterdir()):
        for _, v in enumerate(fileName.iterdir()):
            if v.is_dir():
                for _, fileName in enumerate(v.iterdir()):
                    if fileName.name.split('_')[1] == '影像.tif':

                        newPath = sourcePath_image / fileName.name.split('_')[0] + '_sat.tif'
                        fileName.replace(newPath)
                    elif fileName.name.split('_')[1] == '高程.tif':
                        newPath = sourcePath_dem / fileName.name.split('_')[0] + '_dem.tif'
                        fileName.replace(newPath)

def getResult():
    """
    将dem数据与遥感数据缩放到并裁剪到固定大小（同box大小相同)
    数据类型:
        dem：.tif格式，大小152*152, 168*151等, 精度8.5米/像素
        遥感数据：.tif格式，大小为2560*2560， 2304*2304, 2304*2560等， 精度0.53米/像素
    """
    targetWith_image = targetSize_image[0]
    targetHeight_image = targetSize_image[1]
    targetWidth_dem = targetSize_dem[0]
    targetHeight_dem = targetSize_dem[1]
    box_image = (0, 0, targetHeight_image, targetWith_image)    # 模板
    box_dem = (0, 0, targetHeight_dem, targetWidth_dem)          # 模板
    
    image_list = []  #遥感数据文件名（不含后缀）
    dem_list = []   # dem文件名（不含后缀）
    # 检查dem与遥感数据是否匹配
    for _, v in enumerate(sourcePath_dem.iterdir()):
        dem_list.append(v.stem)
    for _, v in enumerate(sourcePath_image.iterdir()):
        image_list.append(v.stem)
    if not dem_list==image_list:
        raise ValueError('dem与遥感数据不匹配，请检查！')
    # 改变数据大小
    for _, fileName in enumerate(image_list):
        print(fileName)
        jpeg_file = Image.open(sourcePath_image / (fileName + '.tif')).convert('RGB')
        dem_file = Image.open(sourcePath_dem / (fileName + '.tif'))
        img_width = jpeg_file.width
        img_height = jpeg_file.height
        dem_width = dem_file.width
        dem_height = dem_file.height
        dem_file = dem_file.resize((img_width, img_height))
        # result = dem2bmp(dem_file)
        # result.save('DEMImages/' + fileName + '_dem.jpg', quality=95)

        # 宽高均大于目标大小，裁剪图片
        if img_width > targetWith_image and img_height > 2448:
            jpeg_file = jpeg_file.crop(box_image)
            dem_file = dem_file.crop(box_dem)
        # 宽高均小于目标大小，resize图片
        elif img_width < targetWith_image and img_height < targetHeight_image:
            jpeg_file = jpeg_file.resize((targetWith_image, targetHeight_image))
            dem_file = dem_file.resize((targetWidth_dem, targetHeight_dem))
        # 一边大于目标大小，一边小于目标大小，将大的一头先裁剪同小的一样，再resize   
        else:
            min_edges = (dem_height, img_height) if img_width > img_height else (dem_width, img_width)
            dem_file = dem_file.crop((0, 0, min_edges[0], min_edges[0]))
            dem_file = dem_file.resize((targetWidth_dem, targetHeight_dem))
            jpeg_file = jpeg_file.crop((0, 0, min_edges[1], min_edges[1]))
            jpeg_file = jpeg_file.resize((targetWith_image, targetHeight_image))

        remote_jpg = jpeg_file
        dem_tif = dem_file
        # 保存文件
        remote_jpg.save(targetPath_image / (fileName + '.jpg'), quality=95)
        dem_tif.save(targetPath_dem / (fileName + '.tif'), quality=95)


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

def clean():
    remoteData = Path.cwd() / 'original/remoteData'
    demData = Path.cwd() / 'original/demData'
    dem_list = []
    for fileName in demData.iterdir():
        dem_list.append(fileName.stem)
    for file in remoteData.iterdir():
        if file.stem not in dem_list:
            file.unlink()

if __name__ == "__main__":
    # getResult()
    # source = Path('mountain')
    # target = Path('SegmentationClass')
    # result2label(source, target)
    img = Image.open('original/ElevationData/bagui.tif')
    image = np.array(img)
    print(np.min(image), np.max(image))
