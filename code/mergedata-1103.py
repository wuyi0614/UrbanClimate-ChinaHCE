import pandas as pd
import os as os
if __name__ == '__main__':
    file_root = "../data/energyuse-1103.csv"
    readFile = pd.read_csv(file_root, encoding='utf8').reset_index()
    os.getcwd()
    output_df = readFile
    output_df['type'] = 'en_' + output_df['type']
    output_df_group = output_df.groupby(['id','type'])['use'].sum().reset_index()
    output_df_group = output_df_group.pivot(index="id",
                                            columns="type",
                                            values="use").reset_index()
    readFinalFile = output_df_group
    outputpath = "../data/energyuse-1103-aftergrouped.csv"
    readFinalFile.to_csv(outputpath, sep=',', index=True, header=True)




    cwd = os.getcwd()
    EnergyuseFile_root = "../data/energyuse-1103-aftergrouped.csv"
    readEnergyuseFile = pd.read_csv(EnergyuseFile_root, encoding='gbk')
    BasicDataB4MergeFile_root = "../data/mergedata-1103-B4merge.csv"
    readBasicDataB4MergeFile = pd.read_csv(BasicDataB4MergeFile_root, encoding='gbk')
    data_aftermerge = pd.merge(left=readBasicDataB4MergeFile, right=readEnergyuseFile, on='id', how='left')
    # fill na value with 0
    data_aftermerge_nafill = data_aftermerge
    na_fill_var = ['en_ac','en_computer','en_cooking','en_freezing','en_heating','en_laundry','en_lighting','en_television','en_vehicle','en_waterheating']
    data_aftermerge_nafill[na_fill_var] = data_aftermerge_nafill[na_fill_var].fillna(0)
    # print(dataForAnalysis1031_aftermerge)

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

    data_aftermerge['en_total'] = data_aftermerge['en_ac'] + data_aftermerge['en_computer'] + data_aftermerge['en_cooking'] + data_aftermerge['en_freezing'] + data_aftermerge['en_heating'] + data_aftermerge['en_laundry'] + data_aftermerge['en_lighting'] + data_aftermerge['en_television'] + data_aftermerge['en_vehicle'] + data_aftermerge['en_waterheating']
    data_aftermerge['en_total_no_vehicle'] = data_aftermerge['en_ac'] + data_aftermerge['en_computer'] + data_aftermerge[
        'en_cooking'] + data_aftermerge['en_freezing'] + data_aftermerge['en_heating'] + data_aftermerge['en_laundry'] + \
                                  data_aftermerge['en_lighting'] + data_aftermerge['en_television'] + data_aftermerge['en_waterheating']

    data_aftermerge['en_total_percap'] = data_aftermerge[
                                                                        'en_total'] / \
                                                                    data_aftermerge['size']


    outputpath = "../data/mergedata-1103.csv"

    data_aftermerge.to_csv(outputpath, sep=',', index=False, header=True)





