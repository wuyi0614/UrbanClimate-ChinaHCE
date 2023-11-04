import pandas as pd
import xlrd
import os as os
if __name__ == '__main__':
    cwd = os.getcwd()
    regionFile_root = "../data/dataRegionforMerge.csv"
    readregionFile = pd.read_csv(regionFile_root, encoding='utf-8')
    BasicDataB4MergeFile_root = "../data/mergedata-1031.csv"
    readBasicDataB4MergeFile = pd.read_csv(BasicDataB4MergeFile_root, encoding='utf-8')
    dataAftermerge = pd.merge(left=readBasicDataB4MergeFile, right=readregionFile, on='id', how='left')

    outputpath = "../data/mergedata-1101.csv"
    dataAftermerge.to_csv(outputpath, sep=',', index=False, header=True)

