# The main script for the project "Household-level model & prediction"
#
# Create
# d by Mario at 2020-12-12.
#

from pathlib import Path
from config import *

if __name__ == '__main__':
    import pandas as pd

    # load data table
    # 2014: 3863 lines, 1003 columns
    # 2013: ...
    # 2012: ...
    sheet = Path('data') / 'CGSS-unprocessed-202302.xlsx'
    raw = pd.read_excel(sheet, engine='openpyxl', header=[0], skiprows=[1])

    # convert single questions
    features = {
        'a2': 'gender',  # 性别
        'a3_1': 'age',  # 受访者年龄
        'a11': 'house_area',  # 居住面积
        'a18': 'resident',  # 户口状态, 1=城市; 0=农村
        'a29': 'media',  # 在以上媒体中，哪个是您最主要的信息来源？
        'a32': 'outside',  # 在过去一年中，您有多少个晚上是因为出去度假或者探亲访友而没有在家过夜？
        'a33': 'aware_trust',  # 总的来说，您同不同意在这个社会上，绝大多数人都是可以信任的？
        'a34': 'aware_harm',  # 总的来说，您同不同意在这个社会上，您一不小心，别人就会想办法占您的便宜？
        'a35': 'aware_justice',  # 总的来说，您认为当今的社会公不公平？
        'a36': 'aware_happy',  # 总的来说，您觉得您的生活是否幸福？
        'a58': 'job',  # 您的工作经历及状况是？
        'a62': 'income',  # 您家2014年全年家庭总收入是多少？ (类别化)
        'a62_raw': 'raw_income',  # 未处理过的连续收入数据
        'a63': 'size',  # 您家目前住在一起的通常有几个人？
        'e76_1': 'expenditure',  # [家庭总支出] 2014年您全家的家庭支出情况是：
        'e2': 'live_days',  # 您家2014年平均每周在该住房居住时长是：
        'inc': 'income_percap'  # 家庭人均收入
    }

    _, a2 = converter_a2(raw.loc[:, 'a2'])
    _, a3_1 = converter_a3_1(raw.loc[:, 'a3_1'])
    _, a11 = converter_a11(raw.loc[:, 'a11'])
    _, a18 = converter_a18(raw.loc[:, 'a18'])
    _, a29 = converter_a29(raw.loc[:, 'a29'])
    _, a32 = converter_a32(raw.loc[:, 'a32'])
    _, a33 = converter_a33(raw.loc[:, 'a33'])
    _, a34 = converter_a34(raw.loc[:, 'a34'])
    _, a35 = converter_a35(raw.loc[:, 'a35'])
    _, a36 = converter_a36(raw.loc[:, 'a36'])
    _, a58 = converter_a58(raw.loc[:, 'a58'])
    _, a62 = converter_a62(raw.loc[:, 'a62'])
    _, a62_raw = converter_a62_raw(raw.loc[:, 'a62'])
    _, a63 = converter_a63(raw.loc[:, 'a63'])
    _, e76_1 = converter_e76_1(raw.loc[:, 'e76_1'])
    _, inc = converter_income_percap(raw.loc[:, ['a62', 'a63']])
    _, e2 = converter_e2(raw.loc[:, 'e2'])

    # convert array-sized questions
    array_features = {
        'a28': 'media_use',  # [1. 报纸] 过去一年，您对以下媒体的使用情况是：
        'a30': 'recreation',  # [1. 看电视或者看碟] 过去一年，您是否经常在空闲时间从事以下活动？ 休闲活动
        'a31': 'social',  # [1. 社交/串门] 在过去一年中，您是否经常在您的空闲时间做下面的事情？
        'a0101': 'fam_age',  # [14]年龄
        'a0103': 'fam_live',  # [{A0101_1_1}] 请问他们目前是否与您同吃同住？
        'a0104': 'fam_eco',  # [{A0101_1_1}] 请问他们经济上是否与您独立？
        'a0105': 'fam_marriage',  # [{A0101_1_1}] 请问他们的婚姻状况是？
        'a0106': 'fam_relationship'  # [您{A0101_1_1}是您的] 请问他们是您的？
    }

    # a28_keys = [f'a28_{i}' for i in range(1, 7, 1)]
    # _, a28_array = converter_a28_array(raw.loc[:, a28_keys])
    # a30_keys = [f'a30_{i}' for i in range(1, 13, 1)]
    # _, a30_array = converter_a30_array(raw.loc[:, a30_keys])
    # a31_keys = [f'a31_{i}' for i in range(1, 4, 1)]
    # _, a31_array = converter_a31_array(raw.loc[:, a31_keys])
    #
    # a0101_keys = [f'a0101_{i}_2' for i in range(1, 15, 1)]
    # _, a0101_array = converter_a0101_array(raw.loc[:, a0101_keys])
    # a0103_keys = [f'a0103_{i}' for i in range(1, 15, 1)]
    # _, a0103_array = converter_a0103_array(raw.loc[:, a0103_keys])
    # a0104_keys = [f'a0104_{i}' for i in range(1, 15, 1)]
    # _, a0104_array = converter_a0104_array(raw.loc[:, a0104_keys])
    # a0105_keys = [f'a0105_{i}' for i in range(1, 15, 1)]
    # _, a0105_array = converter_a0105_array(raw.loc[:, a0105_keys])
    # a0106_keys = [f'a0106_{i}' for i in range(1, 15, 1)]
    # _, a0106_array = converter_a0106_array(raw.loc[:, a0106_keys])

    # cast a new feature dataframe
    fdf = raw.loc[:, ['id', 'province', 'prefecture', 'county']]
    for f, label in features.items():
        col = pd.DataFrame(locals()[f]).fillna(-99)
        col.columns = [label]
        fdf = pd.concat([fdf, col], axis=1)

    # for f, label in array_features.items():
    #     col = locals()[f'{f}_array'].fillna(-99)
    #     labels = [f'{label}{i}' for i in range(1, col.shape[1] + 1, 1)]
    #     col.columns = labels
    #     fdf = pd.concat([fdf, col], axis=1)

    final = fdf[~fdf.id.isna()]

    # 电器特征抽取
    # 炊事设备
    final['num_cooking'] = converter_e12(raw.loc[:, 'e12'])
    final['power_cooking'] = converter_e13141516c(raw.loc[:, [f'e{i}c' for i in range(13, 17, 1)]])
    final['freq_cooking'] = converter_e13141516d(raw.loc[:, [f'e{i}d' for i in range(13, 17, 1)]])
    final['time_cooking'] = converter_e13141516e(raw.loc[:, [f'e{i}e' for i in range(13, 17, 1)]])

    # 热水器
    final['num_water_heater'] = converter_e12(raw.loc[:, 'e56'])
    final['freq_water_heater'] = converter_e13141516d(raw.loc[:, [f'e{i}e' for i in (57, 58)]])
    final['time_water_heater'] = converter_e13141516e(raw.loc[:, [f'e{i}f' for i in (57, 58)]])
    final['label_water_heater'] = converter_e5758g(raw.loc[:, [f'e{i}g' for i in (57, 58)]])

    # 空调
    final['num_ac'] = converter_e12(raw.loc[:, 'e56'])
    final['freq_ac'] = converter_e60616263ge42(raw.loc[:, [f'e{i}g' for i in range(60, 64, 1)]])
    final['power_ac'] = converter_e60616263d(raw.loc[:, [f'e{i}d' for i in range(60, 64, 1)]])
    final['time_ac'] = converter_e13141516e(raw.loc[:, [f'e{i}h' for i in range(60, 64, 1)]])
    final['label_ac'] = converter_e5758g(raw.loc[:, [f'e{i}e' for i in range(60, 64, 1)]])

    # 供暖
    final['type_heating'] = converter_e39(raw.loc[:, 'e39'])
    final['time_heating'] = converter_e60616263ge42(raw.loc[:, ['e42']])
    final['area_heating'] = converter_e44(raw.loc[:, 'e44'])
    final['cost_heating'] = converter_e50_2e74_1(raw.loc[:, 'e50_2'])

    # 交通工具
    final['own_vehicle'] = converter_e64(raw.loc[:, 'e64'])
    final['emit_vehicle'] = converter_e67e72e73(raw.loc[:, 'e67'])
    final['fuel_vehicle'] = converter_e67e72e73(raw.loc[:, 'e72'])        # 2014实际百公里油耗
    final['fuel_price_vehicle'] = converter_e67e72e73(raw.loc[:, 'e73'])  # 2014平均燃料价格
    final['cost_vehicle'] = converter_e50_2e74_1(raw.loc[:, 'e74_1'])
    final.to_excel(Path('data') / 'vardata-1030.xlsx', index=False)
