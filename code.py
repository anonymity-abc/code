
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:13:38 2023

@author: Administrator
"""

# !/usr/bin/env python
# coding: utf-8

import xgboost as XGBR
#import xgboost as x
import numpy as np
import pandas as pd
from osgeo import gdal
import os
import sys
import time


pco2 = XGBR()
pco2.load_model("XGBOOT.json")##  



FilePath = r"H:\data\2015"  
file_names = os.listdir(FilePath)
SavePath = r"H:\data\result2015"  ###


def image_open(img):
    data = gdal.Open(img)
    if data == "None":
        print("no")
    return data


start_time = time.time()  

k = 0
n = 0
for m in file_names:
    k = k + 1
    father_file_path = FilePath + "/" + m

    new_file_path = SavePath + "/" + m + "pco2_result"
    if os.path.exists(new_file_path):  
        continue
    else:
        os.mkdir(new_file_path)  

    try:
        son_file_names = os.listdir(father_file_path)
    except NotADirectoryError:
        print("no folder")
        break
    else:
        if son_file_names == []:
            print("folder " + m + " empty")
            print("--------------------------------------------------------------------------------------")
            continue
        else:
            print("loading " + m)
            j = 0
            for i in son_file_names:  # 
                QZ = os.path.splitext(i)[0]  # 
                HZ = os.path.splitext(i)[1]  # 
                if (HZ == ".tif"):
                    j = j + 1
                    image = father_file_path + "/" + i
                
                    data = image_open(image)
                    Coastal = data.GetRasterBand(1).ReadAsArray().astype(np.float32)
                    Blue = data.GetRasterBand(2).ReadAsArray().astype(np.float32)
                    Green = data.GetRasterBand(3).ReadAsArray().astype(np.float32)
                    Red = data.GetRasterBand(4).ReadAsArray().astype(np.float32)
                    Nir = data.GetRasterBand(5).ReadAsArray().astype(np.float32)
                    Swir = data.GetRasterBand(6).ReadAsArray().astype(np.float32)
                    Swir1 = data.GetRasterBand(7).ReadAsArray().astype(np.float32)

                
                    Rrs_data = pd.DataFrame({'Coastal':Coastal.reshape(-1),
                                             'Blue':Blue.reshape(-1),
                                             'Green':Green.reshape(-1),
                                             'Red':Red.reshape(-1),
                                             'Nir':Nir.reshape(-1),
                                             'Swir':Swir.reshape(-1),
                                             'Swir1':Swir1.reshape(-1)})
                    Rrs_data_cpu = Rrs_data.multiply(0.0000275).add(-0.2)

                    B1 = Rrs_data_cpu.loc[:, 'Coastal'].values.reshape(-1)
                    B2 = Rrs_data_cpu.loc[:, 'Blue'].values.reshape(-1)
                    B3 = Rrs_data_cpu.loc[:, 'Green'].values.reshape(-1)
                    B4 = Rrs_data_cpu.loc[:, 'Red'].values.reshape(-1)
                    B5 = Rrs_data_cpu.loc[:, 'Nir'].values.reshape(-1)
                    B6 = Rrs_data_cpu.loc[:, 'Swir'].values.reshape(-1)
                    B7 = Rrs_data_cpu.loc[:, 'Swir1'].values.reshape(-1)

               
                    B1[Mask] = 1
                    B2[Mask] = 2
                    B3[Mask] = 1
                    B4[Mask] = 0
                    B5[Mask] = 1
                    B6[Mask] = 1
                    B7[Mask] = 1
      

                    pco2_X = pd.DataFrame({"B3-B2": (B3 - B2), "B5-B3": (B5 - B3), "B5-B4": (B5 - B4),	"B52-B52": (B5 - B2)/(B5 + B2),"B54-B54": (B5-B4)/(B5+B4)
})  

                    pco2_input = pco2_X.values.reshape(-1, 5)  
                    pco2_result = pco2.predict(pco2_input).reshape(-1)



                    pco2_Result = pco2_Result.values

                    
                    band1 = pco2_Result.reshape(Blue.shape)

              
                    A_output = gdal.GetDriverByName("GTiff")  
                    output_0 = A_output.Create(new_file_path + "/" + QZ + "_Chla_result.tif", band1.shape[1],
                                               band1.shape[0], bands=1, eType=gdal.GDT_Float32)
                    output_0.SetProjection(data.GetProjection())
                    output_0.SetGeoTransform(data.GetGeoTransform())

                    band1 = output_0.GetRasterBand(1).WriteArray(band1)
                    output_0 = None
                    print(image + " 已完成，这是 " + m + " 文件夹中的第 " + str(j) + " 个文件")

            n = n + j

    print("文件夹 " + m + " 目录下所有结果已生成,这是第 " + str(k) + " 个文件夹")

print("--------------------------------------------------------------------------------------")
print("所有文件夹已处理完毕！")
end_time = time.time() 
print("文件夹总数为：" + str(k))
print("处理文件个数为：" + str(n))
print("处理时间:%d" % (end_time - start_time)) 
sys.exit()


