import pandas as pd
import os as os
if __name__ == '__main__':
    cwd = os.getcwd()
    EnergyuseFile_root = "../data/demo_familysize_regioncode.csv"
    readEnergyuseFile = pd.read_csv(EnergyuseFile_root, encoding='gbk')
    BasicDataB4MergeFile_root = "../data/mergedata-1104-B4merge.csv"
    readBasicDataB4MergeFile = pd.read_csv(BasicDataB4MergeFile_root, encoding='gbk')
    data_aftermerge = pd.merge(left=readBasicDataB4MergeFile, right=readEnergyuseFile, on='id', how='left')


    data_aftermerge['en_ac_percap'] = data_aftermerge['en_ac'] / data_aftermerge['size']
    data_aftermerge['en_computer_percap'] = data_aftermerge['en_computer'] / data_aftermerge['size']
    data_aftermerge['en_cooking_percap'] = data_aftermerge[
                                               'en_cooking'] / \
                                           data_aftermerge['size']
    data_aftermerge['en_freezing_percap'] = data_aftermerge[
                                                'en_freezing'] / \
                                            data_aftermerge['size']
    data_aftermerge['en_heating_percap'] = data_aftermerge[
                                               'en_heating'] / \
                                           data_aftermerge['size']
    data_aftermerge['en_laundry_percap'] = data_aftermerge[
                                               'en_laundry'] / \
                                           data_aftermerge['size']
    data_aftermerge['en_lighting_percap'] = data_aftermerge[
                                                'en_lighting'] / \
                                            data_aftermerge['size']
    data_aftermerge['en_television_percap'] = data_aftermerge[
                                                  'en_television'] / \
                                              data_aftermerge['size']
    data_aftermerge['en_vehicle_percap'] = data_aftermerge[
                                               'en_vehicle'] / \
                                           data_aftermerge['size']
    data_aftermerge['en_waterheating_percap'] = data_aftermerge[
                                                    'en_waterheating'] / \
                                                data_aftermerge['size']

    data_aftermerge['en_total_no_vehicle_percap'] = data_aftermerge[
                                                                        'en_total_no_vehicle'] / \
                                                                    data_aftermerge['size']

    outputpath = "../data/mergedata-1104.csv"
    data_aftermerge.to_csv(outputpath, sep=',', index=False, header=True)

    outputpath = "../data/mergedata-1104.xlsx"
    datafile = "../data/mergedata-1104.csv"
    data = pd.read_csv(datafile, encoding='utf-8')
    data.to_excel(outputpath)

    # outputpath = "C:/Users/左一/Desktop/HCEs_Emily_Polish/pysurvey/data/dataNew/dataForAnalysis1031_aftermerge.csv"
    # #outputpath = os.path.join(cwd,'data/dataNew/dataForAnalysis1031_aftermerge.csv')
    # dataForAnalysis1031_aftermerge.to_csv(outputpath, sep=',', index=True, header=True)
    # outputpath = "C:/Users/左一/Desktop/HCEs_Emily_Polish/pysurvey/data/dataNew/mergedata-1031.xlsx"
    #
    # from pathlib import Path
    #
    # datafile = "../dataNew/mergedata-1031.csv"
    # data = pd.read_csv(datafile, encoding='utf-8')
    # data.to_excel(outputpath)