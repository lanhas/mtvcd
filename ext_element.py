import sys
import numpy as np
from PIL import Image
from pathlib import Path
from labelme import utils
from func import *

sourcePath_land = Path.cwd() / 'optimize/landClassify'

resPath_Segm = Path.cwd() / 'optimize/SegmentationClass'
resPath_village = Path.cwd() / 'optimize/village'
resPath_mountain = Path.cwd() / 'optimize/mountain'
resPath_water = Path.cwd() / 'optimize/water'
resPath_forest = Path.cwd() / 'optimize/forest'
resPath_farm = Path.cwd() / 'optimize/farm'

tempPath_village = Path.cwd() / 'optimize/village_temp'
tempPath_mountain = Path.cwd() / 'optimize/mountain_temp'
tempPath_water = Path.cwd() / 'optimize/water_temp'
tempPath_forest = Path.cwd() / 'optimize/forest_temp'
tempPath_farm = Path.cwd() / 'optimize/farm_temp'

temp_path = Path.cwd() / 'optimize/temp'

def element_extract():
    """
    将土地分类结果中的林、田进行提取，分别保存到指定目录
    """
    for _, fileName in enumerate(sourcePath_land.iterdir()):
        name = fileName.name
        print(name)
        image_land = np.array(Image.open(fileName), np.uint8)

        # image_village = np.zeros_like(image_land, np.uint8)
        # image_water = np.zeros_like(image_land, np.uint8)
        image_forest = np.zeros_like(image_land, np.uint8)
        image_farm = np.zeros_like(image_land, np.uint8)
        
        # image_water[image_land==4] = 1     # 水
        image_forest[image_land==2] = 1     # 林
        image_farm[image_land==3] = 1     # 田
        # image_village[image_land==6] = 1   # 村落
        
        # utils.lblsave(tempPath_water / name, image_water)
        utils.lblsave(tempPath_forest / name, image_forest)
        utils.lblsave(tempPath_farm / name, image_farm)
        # utils.lblsave(tempPath_village / name, image_village)
        # utils.lblsave(temp_path / name, image_water)

def element_merge():
    """
    将多个要素合并起来并保存
    """
    for _, fileName in enumerate(resPath_mountain.iterdir()):
        name = fileName.name
        print(name)
        image_village = np.array(Image.open(resPath_village / name), np.uint8)
        image_mountain = np.array(Image.open(resPath_mountain / name), np.uint8)
        image_water = np.array(Image.open(resPath_water / name), np.uint8)
        image_forest = np.array(Image.open(resPath_forest / name), np.uint8)
        image_farm = np.array(Image.open(resPath_farm / name), np.uint8)

        result = np.zeros_like(image_village, np.uint8)

        result[image_forest==1] = 2     # 林（绿色）
        result[image_farm==1] = 3       # 田（黄色）
        result[image_village==2] = 5    # 荒地（紫色）
        result[image_village==3] = 0    # 未知区域，云、未获取的遥感影像等（黑色）
        result[image_mountain==1] = 1   # 山（红色）
        result[image_village==1] = 6    # 村落（浅蓝色）
        result[image_water==1] = 4      # 水（深蓝色）
        result[result==0] = 2
        utils.lblsave(resPath_Segm / name, result)

def element_adjust(source_path, target_path):
    """
    对要素进行调整，去除边界毛糙部分和小区域
    """
    for _, fileName in enumerate(source_path.iterdir()):
        name = fileName.name
        print(name)
        image = np.array(Image.open(fileName), np.uint8)
        image = boundary_pruning(image, 10, 2)
        utils.lblsave(target_path / name, image)

def change_name(path, separator, suffix):
    """
    更改路径下文件的名字
    Parameters
    ----------
    path: pathlib.Path
        文件夹路径
    separator: str
        需要分隔的符号，example: '_'
    suffix: str
        新文件的后缀，example: '.jpg'
    """
    for _, fileName in enumerate(path.iterdir()):
        name = fileName.name
        newName = path / (name.split(separator)[0] + suffix)
        fileName.rename(newName)

def delete_file(path, nameList):
    """
    递归删除文件
    Parameters
    ----------
    path: pathlib.Path
        文件夹路径
    nameList: list
        需要删除的文件名列表，example：['baiba', 'bagui']
    """
    for fileName in path.iterdir():
        if fileName.is_dir():
            delete_file(fileName, nameList)
        else:
            if fileName.name.split('.')[0] in nameList:
                fileName.unlink()
                print('delete file {}'.format(fileName.name))

def temp():
    for idx, fileName in enumerate(tempPath_mountain.iterdir()):
        name = fileName.name
        print(name)
        image = np.array(Image.open(fileName), np.uint8)
        result = np.zeros_like(image, np.uint8)
        result[image==0] = 1
        utils.lblsave(resPath_mountain / name, result)

def temp1():
    for idx, fileName in enumerate(sourcePath_land.iterdir()):
        name = fileName.name
        print(name)
        image = np.array(Image.open(fileName), np.uint8)
        result = color2annotation(image)
        utils.lblsave(sourcePath_land / name, result)

def replenish():
    current_img = []
    for idx, fileName in enumerate(tempPath_water.iterdir()):
        name = fileName.name
        current_img.append(name)
    for idx, filename in enumerate(sourcePath_land.iterdir()):
        name = filename.name
        if name not in current_img:
            path = resPath_water / name
            image = np.zeros((2448, 2448), dtype=np.uint8)
            utils.lblsave(path, image)


if __name__ == "__main__":
    # nameList = ['daqiao', 'gaopoping', 'guiliu', 'jingbang', 'miliang', 'shengkou']
    # path = Path(r'optimize')
    # delete_file(path, nameList)
    # change_name(path, '.jpg')
    element_merge()
    # element_extract()
    # element_adjust(tempPath_water, resPath_water)
    # element_merge()
    # image = Image.open('optimize/SegmentationClass/bagui.png')
    # image = np.array(image)
    # print(image.max())
    # for idx, fileName in enumerate(resPath_Segm.iterdir()):
    #     name = fileName.name
    #     # print(name)
    #     image = np.array(Image.open(fileName), np.uint8)
    #     print(image.max())



