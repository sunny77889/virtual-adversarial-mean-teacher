# coding=utf-8   
import os
import random
import shutil
import sys
import struct
#for python3
from functools import cmp_to_key

FILE_DIR = sys.argv[1]
FILE_MAX_NUM = 60000  
dirName = ''
PNG_SIZE = 784
def processSession(path):  
    global allFileNum  
    root = os.getcwd()
    outDir = ''
    global dirName
    # 所有文件夹，第一个字段是次目录的级别  
    dirList = []  
    # 所有文件  
    fileList = []  
    # 返回一个列表，其中包含在目录条目的名称
    files = os.listdir(path)  
    for f in files:  
        if(os.path.isdir(path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
            if(f[0] == '.'):  
                pass  
            else:  
                # 添加非隐藏文件夹  
                dirList.append(f)  
        if(os.path.isfile(path + '/' + f)):  
            # 添加文件  
            fileList.append(f)  
    # 当一个标志使用，文件夹列表第一个级别不打印  
    for dl in dirList:  
        dirName = dl  
        processSession(path + '/' + dl)
    fileListLength = len(fileList)
    if fileListLength != 0:
        # fileList.sort(cmp = lambda x,y : int(os.path.getsize(path + '/' + x) - os.path.getsize(path + '/' + y)),
        # reverse = True) 
        fileList.sort(key = cmp_to_key(lambda x,y : int(os.path.getsize(path + '/' + x) - os.path.getsize(path + '/' + y))),
        reverse = True) 
        if fileListLength > FILE_MAX_NUM:
            fileListLength = FILE_MAX_NUM
            fileList = fileList[:FILE_MAX_NUM]
        trainSet = random.sample(fileList,fileListLength - fileListLength//10)
        testSet = list(set(fileList) - set(trainSet))
        #for 10%
        # print(9*len(trainSet)//100)
        trainSet = random.sample(trainSet,len(trainSet)//100)
        for fl in trainSet:
            # print fl,os.path.getsize(path + '/' + fl)
            outDir = root + '/3_ProcessedSession/Train/' + dirName
            if os.path.exists(outDir) == False:
                os.makedirs(outDir)
            trimedFile(path + '/' + fl,outDir +'/'+fl)
        for fl in testSet:
            # print fl,os.path.getsize(path + '/' + fl)
            testDir = root + '/3_ProcessedSession/Test/' + dirName
            if os.path.exists(testDir) == False:
                os.makedirs(testDir)
            trimedFile(path + '/' + fl,testDir +'/'+fl)
def trimedFile(srcPath,outPath):
    with open(srcPath,'rb') as fi:
        byte = fi.read(PNG_SIZE)
        while len(byte) < PNG_SIZE:
            byte = byte + struct.pack('b',0)
        with open(outPath,'wb') as fo:
            fo.write(byte)
    pass
if __name__ == '__main__':  
    processSession(FILE_DIR)