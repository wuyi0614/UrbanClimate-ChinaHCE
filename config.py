# 此处可导入和变量转换 Mapping 的具体函数和参数
# 创建于 2021-11-03
#
import re
import pandas as pd
import numpy as np

from collections import Counter


# 1. 数据变量的数值转换
# 以下为2014年CGSS数据表的变量转换（注意统一转换函数的写法，确保在运行的时候代码是一致的）
# 另外要注意的是，数据表里，表头下一行是问题的说明，并非数据要避开

# a5_01, a5_11, a5_12, a5_13, a5_14, a5_15, a5_16, a5_17, a5_18, a5_19, a5_20
def binary(series: pd.Series):
    mapping = {"是": 1, "否": 0}
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


# a11 您现在住的这座住房的套内建筑面积是：
def converter_a11(series: pd.Series):
    mapping = (
        ([0, 20], 0),
        ([21, 50], 1),
        ([51, 100], 2),
        ([101, 300], 3),
        ([301, 500], 4),
        ([501, 10000], 5)
    )

    def _convert(area):
        area = int(area)
        label = -99
        for scale, v in mapping:
            if scale[0] <= area <= scale[1]:
                label = v
            else:
                continue
        return label

    array = series.apply(_convert)
    return mapping, array


def converter_a18(series: pd.Series):
    mapping = {
        "非农业户口": 1,
        "农业户口": 0,
        "居民户口（以前是非农业户口）": 1,
        "居民户口（以前是农业户口）": 1,
        "军籍": 1,
        "没有户口": 0,
        "其他": 0
    }
    array = series.apply(lambda x: mapping.get(x, 0))
    return mapping, array


# 在以上媒体中，哪个是您最主要的信息来源？
def converter_a29(series: pd.Series):
    series = series.fillna(-99)
    series = series.astype(int)  # should be int
    return None, series.values


# 在过去一年中，您有多少个晚上是因为出去度假或者探亲访友而没有在家过夜？
def converter_a32(series: pd.Series):
    mapping = {
        "无法回答": 0,
        "从未": 1,
        "1-5个晚上": 2,
        "6-10个晚上": 3,
        "11-20个晚上": 4,
        "21-30个晚上": 5,
        "超过30个晚上": 6
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


# 总的来说，您同不同意在这个社会上，绝大多数人都是可以信任的？
def converter_a33(series: pd.Series):
    mapping = {
        "无法回答": 0,
        "非常不同意": 1,
        "比较不同意": 2,
        "说不上同意不同意": 3,
        "比较同意 ": 4,
        "非常同意": 5
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a34(series: pd.Series):
    mapping = {
        "无法回答": 0,
        "非常不同意": 1,
        "比较不同意": 2,
        "说不上同意不同意": 3,
        "比较同意 ": 4,
        "非常同意": 5
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a35(series: pd.Series):
    mapping = {
        "无法回答": 0,
        "完全不公平": 1,
        "比较不公平": 2,
        "说不上公平但也不能说不公平": 3,
        "比较公平": 4,
        "完全公平": 5
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a36(series: pd.Series):
    mapping = {
        "无法回答": 0,
        "非常不幸福": 1,
        "比较不幸福": 2,
        "说不上幸福不幸福": 3,
        "比较幸福": 4,
        "非常幸福": 5
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a7a(series: pd.Series):
    raw_mapping = {"未接受教育": [
        "扫盲班",
        "没有受过任何教育"
    ],
        "初中及以下": [
            "Сѧ",
            "初中"
        ],
        "高中": [
            "普通高中",
            "职业高中",
            "中专"
        ],
        "大学": [
            "大学本科(成人高等教育)",
            "大学本科(正规高等教育)",
            "大学专科(成人高等教育)",
            "大学专科(正规高等教育)"
        ],
        "研究生及以上": [
            "研究生及以上"
        ],
        "其他": [
            "其他",
            "无法回答",
            "技校"
        ]}
    mapping = {}
    for idx, (k, v) in enumerate(raw_mapping.items()):
        for item in v:
            mapping[item] = idx

    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a58(series: pd.Series):
    raw_mapping = {
        "失业": [
            "从未工作过",
            "目前没有工作，曾经有过非农工作",
            "目前没有工作，而且只务过农"
        ],
        "务农工作": [
            "目前务农，没有过非农工作",
            "目前务农，曾经有过非农工作"
        ],
        "非务农工作": [
            "目前从事非农工作"
        ]}

    mapping = {}
    for idx, (k, v) in enumerate(raw_mapping.items()):
        for item in v:
            mapping[item] = idx

    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


# 您家2014年全年家庭总收入是多少？
def converter_a62(series: pd.Series):
    # TODO: 需要重新设计一下，分的不均等可能有问题，还有就是样本分布是不是被cover
    mapping = (
        ([0, 2800], 0),
        ([2801, 10000], 1),
        ([10001, 25000], 2),
        ([25001, 50000], 3),
        ([50001, 100000], 4),
        ([100001, 200000], 5),
        ([200001, 400000], 6),
        ([400001, 10000000], 7)
    )

    def _convert(inc):
        inc = int(inc)
        label = -99
        for scale, v in mapping:
            if scale[0] <= inc <= scale[1]:
                label = v
            else:
                continue
        return label

    array = series.apply(_convert)
    return mapping, array


def converter_a62_raw(series: pd.Series):
    series = series.astype(int)  # should be int
    array = series.apply(lambda x: -99 if x >= 9999990 else x)
    return None, array


# 家庭成员 a63
def converter_a63(series: pd.Series):
    series = series.astype(int)  # should be int
    array = series.apply(lambda x: 1 if x >= 97 else x)
    return None, array


def converter_e76_1(series: pd.Series):
    series = series.astype(int)  # should be int
    array = series.apply(lambda x: -99 if x >= 9999990 else x)
    return None, array


# e2: 您家2014年平均每周在该住房居住时长是：
def converter_e2(series: pd.Series):
    series = series.fillna(-99)
    series = series.astype(int)  # should be int
    array = series.values
    return None, array


def converter_a2(series: pd.Series):
    mapping = {
        "男": 1,
        "女": 0,
        "Ů": -1
    }
    array = series.apply(lambda x: mapping.get(x, -99))
    return mapping, array


def converter_a3_1(series: pd.Series):
    # respondent's age is his born year
    base = 2014
    series = base - series.astype(int)
    return None, series.values


def converter_income_percap(array: pd.DataFrame):
    # calculate income percap based on income and resident size
    # use `a62` and `a63`
    return None, array['a62'].astype(float).values / array['a63'].astype(int).values


def converter_a30_array(data: pd.DataFrame):
    # a30 ranges from 1 - 12
    mapping = {
        "无法回答": 0,
        "从不": 1,
        "一年数次或更少": 2,
        "一月数次": 3,
        "一周数次": 4,
        "每天": 5
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


def converter_a31_array(data: pd.DataFrame):
    # a31 ranges from 1 - 3
    mapping = {
        "从不": 0,
        "有时": 1,
        "很少": 2,
        "经常": 3,
        "非常频繁": 4
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


# [1. 报纸] 过去一年，您对以下媒体的使用情况是 6 options
def converter_a28_array(data: pd.DataFrame):
    mapping = {
        "从不": 0,
        "有时": 1,
        "很少": 2,
        "经常": 3,
        "非常频繁": 4
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


# 家庭成员年龄: 14 options
def converter_a0106_array(data: pd.DataFrame):
    raw_mapping = {
        "配偶": [
            "配偶"
        ],
        "兄弟姐妹": [
            "兄弟姐妹"
        ],
        "父母": [
            "父母",
            "配偶的父母"
        ],
        "子女": [
            "子女",
            "女婿/儿媳"
        ],
        "祖父辈": [
            "曾祖父母/曾外祖父母",
            "祖父母/外祖父母"
        ],
        "孙辈": [
            "孙子(女)/外孙子(女)"
        ],
        "其他": [
            "姑妈(父亲的姐妹)",
            "配偶的其他亲属",
            "其他非亲属",
            "其他亲属"
        ]
    }
    mapping = {}
    for idx, (k, v) in enumerate(raw_mapping.items()):
        for item in v:
            mapping[item] = idx

    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


def converter_a0101_array(data: pd.Series):
    mapping = (([0, 20], 0), ([21, 40], 1), ([41, 60], 2), ([61, 80], 3), ([81, 1000], 4))

    def _convert(series):
        series = series.fillna(-99)
        items = []
        for it in series:
            label = -99
            for scale, v in mapping:
                if scale[0] <= it <= scale[1]:
                    label = v
                else:
                    continue

            items += [label]
        return items

    array = data.apply(_convert, axis=1)
    return mapping, pd.DataFrame(list(array.values))


def converter_a0103_array(data: pd.DataFrame):
    """`_array` means it has 14 options"""
    mapping = {
        "无法回答": 0,
        "吃住都不在一起": 1,
        "住在一起，但吃不在一起": 2,
        "吃在一起，但住不在一起": 3,
        "吃住都在一起": 4,
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


def converter_a0104_array(data: pd.DataFrame):
    mapping = {
        "是": 1,
        "否": 0
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


def converter_a0105_array(data: pd.DataFrame):
    mapping = {
        "已婚": 1,
        "未婚": 0,
        "离婚": 0
    }
    array = data.apply(lambda x: [mapping.get(it, -99) for it in x], axis=1)
    return mapping, pd.DataFrame(list(array.values))


# 构造炊事设备、电器、能源、供暖、私人交通、用电相关的 lifestyle 变量
def _power_process(x):
    if not isinstance(x, str):
        return -99

    searched = re.findall(r'\d+', x)
    if len(searched) == 1:
        if x.startswith('<'):
            p = int(searched.pop()) / 2
        else:
            p = int(searched.pop())

    elif len(searched) == 2:
        p = (int(searched[0]) + int(searched[1])) / 2
    else:
        p = -99
    return p


def _freq_process(x):
    # 单位是 x 次/天，每周的频率/7，每月的频率/30
    if not isinstance(x, str):
        return -99

    searched = re.findall(r'\d+', x)
    if not searched:
        return -99

    else:
        f = int(searched.pop())

    if x.startswith('每天'):
        pass
    elif x.startswith('每周'):
        f = round(f / 7, 3)
    elif '月' in x:  # 每月
        f = round(f / 30, 3)
    else:
        f = -99
    return f


def _time_process(x):
    # 处理一些电器使用的时长变量: [1-2)Сʱ, [15-30)分钟 等
    if x == -99 or not isinstance(x, str):
        return -99

    found = re.findall('\d+', x)
    if 'Сʱ' in x:
        t = sum([float(it) for it in found]) / len(found)
        t = round(t * 60, 3)
    elif '分钟' in x:
        t = sum([float(it) for it in found]) / len(found)
    else:
        t = -99

    return t


def _monthly_use_process(x):
    if x == -99 or not isinstance(x, str):
        return -99

    found = re.findall('\d+', x)
    if '月' in x:
        t = sum([float(it) for it in found]) / len(found)
    else:
        t = -99

    return t


def _eff_label_process(x):
    # 电器能源效率标识，二级、三级、四级、五级，越高越好；没有标识则默认为 1
    if not isinstance(x, str):
        return -99

    if '二级' in x:
        return 2
    elif '三级' in x:
        return 3
    elif '四级' in x:
        return 4
    elif '五级' in x:
        return 5
    elif '没有' in x:
        return 1
    else:
        return -99


def _ac_power_process(x):
    # 空调的功率是，1.5匹机（3.6-4.5kw）一类，最终单位是W
    if not isinstance(x, str):
        return -99

    found = re.findall(r'\d+', x)
    if 'kW' in x:
        p = sum([float(it) for it in found]) / len(found)
        p = p * 1000
    else:
        p = -99

    return p


def _basic_numeric_process(x):
    # 供暖面积，单位是平方米; 百公里油耗，单位是L
    if not isinstance(x, str):
        return -99

    found = re.findall(r'\d+', x)
    if found:
        p = sum([float(it) for it in found]) / len(found)
    else:
        p = -99

    return p


def converter_cooking(data: pd.DataFrame):
    # 一共有四台炊事设备
    da = data.copy(True)
    cnum = da[['e12']].astype(int)
    cnum.columns = ['cooking_num']
    devices = ['e13', 'e14', 'e15', 'e16']

    for de in devices:
        da[f'{de}-power'] = da[f'{de}c'].apply(_power_process)
        da[f'{de}-freq'] = da[f'{de}d'].apply(_freq_process)

    # 尽量将不同的变量都整合到一起，比如计算平均功率，平均使用频率
    cpower = da[[f'{de}-power' for de in devices]].apply(lambda x: x[x>0].mean(), axis=1).fillna(-99)
    cfreq = da[[f'{de}-freq' for de in devices]].apply(lambda x: x[x>0].mean(), axis=1).fillna(-99)
    cnum['cook_power'] = cpower
    cnum['cook_freq'] = cfreq
    return cnum


# 能耗数据的转换和构造
def converter_e12(series):
    # 炊事设备数量; 同样适用于e56
    # 炊事设备 >10 的值视为 10
    series = series.fillna(-99).astype(int).values
    series[series > 10] = 10
    return series


def converter_e13141516c(df: pd.DataFrame):
    # e13c - e16c have all the same structure
    # 功率的单位是 W
    df = df.fillna(-99)  # many of them are empty
    v = df.apply(lambda x: np.array([_power_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e13141516d(df: pd.DataFrame):
    # 频率的单位是 次/天; 同样适用于e57-58e
    df = df.fillna(-99)
    v = df.apply(lambda x: np.array([_freq_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e13141516e(df: pd.DataFrame):
    # 同样适用于 e57-58f, 60-63h
    # 使用时长的单位是 分钟
    df = df.fillna(-99)
    v = df.apply(lambda x: np.array([_time_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e5758g(df: pd.DataFrame):
    # 是否有能源效率标准
    df = df.fillna(-99)
    v = df.apply(lambda x: np.array([_eff_label_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e60616263ge42(df: pd.DataFrame):
    # 空调的制冷时长/供暖时长，单位是：月
    df = df.fillna(-99)
    v = df.apply(lambda x: np.array([_monthly_use_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e60616263d(df: pd.DataFrame):
    # 空调的功率，单位是：W（原来是kW）
    df = df.fillna(-99)
    v = df.apply(lambda x: np.array([_ac_power_process(it) for it in x]), axis=1).values
    series = [round(it[it > -99].mean(), 3) if it[it > -99].size > 0 else -99 for it in v]
    return series


def converter_e39(series):
    # 供暖方式，'集中式供暖', '混合供暖（集中式+自供暖）', '分户自供暖', '没有供暖'
    mapping = {'集中式供暖': 0,
               '混合供暖（集中式+自供暖）': 1,
               '分户自供暖': 2,
               '没有供暖': 3}
    return [mapping.get(it, -99) for it in series.values]


def converter_e44(series):
    # 供暖面积，单位是 m2
    series = series.fillna(-99)
    return series.apply(_basic_numeric_process).values


def converter_e50_2e74_1(series):
    # 承担的集中供暖费用, 元; 汽车燃料费用
    series = series.fillna(-99)
    v = series.astype(int).values
    v[v > 90000] = -99
    return v


def converter_e64(series):
    # 是否拥有汽车
    series = series.fillna(-99)
    mapping = {'是': 1, '否': 0}
    return [mapping.get(it, -99) for it in series.values]


def converter_e67e72e73(series):
    # 汽车排量/百公里油耗，单位是 L；燃料价格，单位是 元/L
    series = series.fillna(-99)
    return series.apply(_basic_numeric_process).values


# 2. 特殊变量的构造
def create_demographic_variable(age, rel_array):
    """家庭结构组成
    单身汉；
    夫妻；
    双亲+1孩；
    双亲+2孩；
    单亲+1孩；
    单亲+2孩；
    5人以上大家庭；
    独居老人；
    老年夫妻
    其他"""
    # 需要用到的数组:
    rel_mapping = {'配偶': 0,
                   '兄弟姐妹': 1,
                   '父母': 2,
                   '配偶的父母': 2,
                   '子女': 3,
                   '女婿/儿媳': 3,
                   '曾祖父母/曾外祖父母': 4,
                   '祖父母/外祖父母': 4,
                   '孙子(女)/外孙子(女)': 5,
                   '姑妈(父亲的姐妹)': 6,
                   '配偶的其他亲属': 6,
                   '其他非亲属': 6,
                   '其他亲属': 6}
    age_mapping = (([0, 20], 0),
                   ([21, 40], 1),
                   ([41, 60], 2),
                   ([61, 80], 3),
                   ([81, 1000], 4))

    # 判断规则: 先构造一列新的标记
    single = lambda x: set(x) == {-99}  # 也是独居老人的条件之一
    couple = lambda x: set(x) == {0, -99}  # 也是老年夫妻的条件之一
    couple_kid_one = lambda x: set(x) == {0, 3, -99} or (set(x) == {2, -99} and Counter(x)[2] > 1)
    couple_kid_two = lambda x: (set(x) == {0, 3, -99} and Counter(x)[3] > 1) or (
                set(x) == {1, 2, -99} and Counter(x)[2] > 1)
    single_kid_one = lambda x: set(x) == {3, -99} or (set(x) == {2, -99} and Counter(x)[2] == 1)
    single_kid_two = lambda x: (set(x) == {3, -99} and Counter(x)[3] > 1) or (
                set(x) == {1, 2, -99} and Counter(x)[2] == 1)
    grand_parent_kids = lambda x: set(x) == {4, -99} or set(x) == {5, -99} or set(x) == {0, 4, -99} or set(x) == {0, 5,
                                                                                                                  -99}
    big_family = lambda x: Counter(x)[-99] <= 10

    single = rel_array.apply(single, axis=1)
    couple = rel_array.apply(couple, axis=1)
    couple_kid_one = rel_array.apply(couple_kid_one, axis=1)
    couple_kid_two = rel_array.apply(couple_kid_two, axis=1)
    single_kid_one = rel_array.apply(single_kid_one, axis=1)
    single_kid_two = rel_array.apply(single_kid_two, axis=1)
    grand_parent_kids = rel_array.apply(grand_parent_kids, axis=1)
    big_family = rel_array.apply(big_family, axis=1)

    # 判断规则：根据年龄判断
    elderly = age.apply(lambda x: x >= 60)
    single_elderly = (elderly & single)
    couple_elderly = (elderly & couple)

    # 赋值
    mapping = dict(
        single=0,
        couple=1,
        couple_kid_one=2,
        couple_kid_two=2,
        single_kid_one=3,
        single_kid_two=3,
        grand_parent_kids=4,
        single_elderly=5,
        couple_elderly=6,
        big_family=7
    )
    other = 7
    relationship = pd.Series([other] * len(age))
    for key, v in mapping.items():
        relationship[locals()[key]] = v

    mapping['other'] = other
    return mapping, relationship


# 构造地区变量: 四个地区
def create_regional_variable(series):
    """构建地区分组
    """
    # mapping = {'coastal': ['北京市', '天津市', '河北省', '山东省', '江苏省', '上海市', '浙江省', '福建省', '广东省'],
    #            'northeastern': ['黑龙江省', '吉林省', '辽宁省'],
    #            'central': ['山西省', '河南省', '湖北省', '湖南省', '安徽省', '江西省'],
    #            'western': ['陕西省', '甘肃省', '宁夏回族自治区', '青海省', '四川省', '云南省', '贵州省', '重庆市', '广西壮族自治区', '内蒙古']}
    mapping = {'northeast': ['黑龙江省', '吉林省', '辽宁省'],
               'beijing-tianjin': ['北京市', '天津市'],
               'north': ['河北省', '山东省'],
               'central': ['河南省', '山西省', '安徽省', '湖南省', '湖北省', '江西省'],
               'central-coast': ['上海市', '浙江省', '江苏省'],
               'south-coast': ['广东省', '福建省', '海南省'],
               'northwest': ['内蒙古', '陕西省', '甘肃省', '宁夏回族自治区', '青海省', '新疆'],
               'southwest': ['四川省', '重庆市', '云南省', '贵州省', '广西壮族自治区']}
    mapping_ = {}
    out = {}
    for idx, (k, v) in enumerate(mapping.items()):
        for item in v:
            mapping_[item] = idx

        out[idx] = k

    func = lambda x: mapping_.get(x, '-99')
    return out, series.apply(func)
