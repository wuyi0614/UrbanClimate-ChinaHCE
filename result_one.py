# Replicable script for the first section of Results
#
# Created on 18 Oct 2023, by Yi
#

import json
import pandas as pd

from pathlib import Path
import matplotlib.pyplot as plt


# env variables
MAPPINGFILE = Path('data') / 'mapping.json'
MAPPING = json.loads(MAPPINGFILE.read_text(encoding='utf8'))


# functions
def get_cities(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform cities in Chinese into English

    :param data: the raw dataframe
    :return: an updated dataframe
    """
    city = MAPPING['city']
    data['prefecture_eng'] = data['prefecture'].apply(lambda x: city.get(x, 'other'))
    return data


def get_percap(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Transform selected data into percap data

    :param data: the raw dataframe
    :param columns: columns for percap transformation
    :return: an updated dataframe
    """
    for c in columns:
        data[f'{c}_percap'] = data[c] / data['size']

    return data


def get_energy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform energy consumption data into a well-shaped dataset

    :param data: the raw energy consumption dataframe
    :return: the pivot-processed dataframe
    """
    data['type'] = data['type'].apply(lambda x: f'en_{x}')
    merged = data[['id', 'type', 'use']].groupby(['id', 'type']).sum().reset_index()
    pivot = merged.pivot(index='id', columns='type', values='use').fillna(0)
    pivot['en_total'] = pivot.iloc[:, 1:].sum(axis=1)  # sum by rows
    sub = ['en_ac', 'en_computer', 'en_cooking', 'en_freezing', 'en_heating',
           'en_laundry', 'en_lighting', 'en_television', 'en_waterheating']
    pivot['en_total_no_vehicle'] = pivot[sub].sum(axis=1)
    return pivot


def analyse(data: pd.DataFrame) -> pd.DataFrame:
    """
    The dataframe should contain `region` column before analysing and the chart has the following parts:
    - mean values by cities - box plot
    - mean values by urban/rural - box plot
    - mean values by north/south - solid lines

    :param data: a pre-processed dataframe
    """
    out = data[['prefecture_eng']]  # append data to the tot
    # energy consumption by city
    data['percap_all'] = data['en_total'] / data['size']

    # data by urban/rural
    mask = data['resident'] == 1  # mask for being urban residents

    # pre-processing before making the chart
    # result one: percap energy use by urban/rural by cities
    fig = plt.figure(figsize=(10, 16))
    for ci, g in data.groupby('prefecture_en'):
        plt.boxplot(g['percap_all'], showfliers=False)
        break

    plt.show()
    return


def create_regional_variable(series):
    """构建地区分组

    首先，从南北地区来看，南方地区的农村家庭能源消费量958.3kgce/年，人
    均能源消费量为334.7kgce/年；北方地区农村家庭能源消费量为1311.8kgce/年，
    人均能源消费量为442kgce/年。北方地区农村家庭能源消费量达南方地区的1.37
    倍，人均能源消费量为1.32 倍
    """
    mapping = {'north': ['北京市', '天津市', '河北省', '山东省', '黑龙江省', '吉林省', '辽宁省', '山西省', '河南省',
                         '陕西省', '甘肃省', '宁夏回族自治区', '青海省', '内蒙古', '新疆'],
               'south': ['江苏省', '上海市', '浙江省', '福建省', '广东省', '四川省', '云南省', '贵州省',
                         '重庆市', '广西壮族自治区', '湖北省', '湖南省', '安徽省', '江西省']}
    # mapping = {'coastal': ['北京市', '天津市', '河北省', '山东省', '江苏省', '上海市', '浙江省', '福建省', '广东省'],
    #            'northeastern': ['黑龙江省', '吉林省', '辽宁省'],
    #            'central': ['山西省', '河南省', '湖北省', '湖南省', '安徽省', '江西省'],
    #            'western': ['陕西省', '甘肃省', '宁夏回族自治区', '青海省', '四川省', '云南省', '贵州省', '重庆市', '广西壮族自治区', '内蒙古']}
    # mapping = {'northeast': ['黑龙江省', '吉林省', '辽宁省'],
    #            'beijing-tianjin': ['北京市', '天津市'],
    #            'north': ['河北省', '山东省'],
    #            'central': ['河南省', '山西省', '安徽省', '湖南省', '湖北省', '江西省'],
    #            'central-coast': ['上海市', '浙江省', '江苏省'],
    #            'south-coast': ['广东省', '福建省', '海南省'],
    #            'northwest': ['内蒙古', '陕西省', '甘肃省', '宁夏回族自治区', '青海省', '新疆'],
    #            'southwest': ['四川省', '重庆市', '云南省', '贵州省', '广西壮族自治区']}
    mapping_ = {}
    out = {}
    for idx, (k, v) in enumerate(mapping.items()):
        for item in v:
            mapping_[item] = idx

        out[idx] = k

    func = lambda x: mapping_.get(x, '-99')
    return out, series.apply(func)


if __name__ == '__main__':
    # merging data and test
    datafile = Path('data') / 'vardata-1025.xlsx'
    data = pd.read_excel(datafile, engine='openpyxl')
    engfile = Path('data') / 'energyuse-1024.xlsx'
    eng = pd.read_excel(engfile, engine='openpyxl')
    eng = get_energy(eng)

    merged = data.merge(eng, on='id', how='left')
    # preprocess
    merged = get_cities(merged)
    merged = get_percap(merged, columns=['en_total'])
    _, merged['region'] = create_regional_variable(merged['province'])
    merged.to_excel(Path('data') / 'mergedata-1027.xlsx', index=False)

    """ check the data distribution (should be attached to the appendix)
                en_total_no_vehicle
    region                     
    0               1003.835610
    1                554.723219
    = 1.81
               en_total
    region             
    0       1210.287436
    1        742.899574
    = 1.63
    """
    diff = merged[['region', 'en_total_no_vehicle']].groupby('region').mean()
    diff = merged[['region', 'en_total']].groupby('region').mean()

    # city-level variables
    cityfile = Path('data') / 'citydata-1029.xlsx'
    city = pd.read_excel(cityfile, header=[0], skiprows=[1])
    i = set(merged.prefecture.values).intersection(set(city.Ctnm.values))
    mask = (city['Ctnm'].isin(i)) & (city['Year'] == 2014)
    city = city.loc[mask, ['Ctnm', 'CtEngcnsmp', 'Ppln_Horgpuye']]  # CtEngcnsmp: 10,000 tce,
    city.columns = ['city', 'energy', 'pop']
    city['eng_percap'] = city['energy'] / city['pop']

    foo = merged[['prefecture', 'en_total', 'size']].groupby('prefecture').sum()
    foo['en_total'] / foo['size'] / 1000
