import pandas as pd
import xlrd
import os as os
if __name__ == '__main__':
    cwd = os.getcwd()
    DemoFile_root = "../data/dataDemo.csv"
    readDemoFile = pd.read_csv(DemoFile_root, encoding='gbk')
    B4mergeFile_root = "../data/mergedata-1030.csv"
    readB4mergeFile = pd.read_csv(B4mergeFile_root, encoding='gbk')
    mergedata1031 = pd.merge(left=readB4mergeFile, right=readDemoFile, on='id', how='left')

    outputpath = "../data/mergedata-1031.csv"
    mergedata1031.to_csv(outputpath, sep=',', index=True, header=True)





