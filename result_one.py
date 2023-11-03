# Replicable script for the first section of Results
#
# Created on 18 Oct 2023, by Yi
#

import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from config import WSJ

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
    data['prefecture_eng'] = data['prefecture'].apply(lambda x: city.get(x, 'other').capitalize())
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


def city_lifestyle_chart(energy: pd.DataFrame) -> pd.DataFrame:
    """
    This function illustrates the components of energy consumption from each category

    :param energy: the energy dataframe
    :return: the used dataframe
    """
    # the chart has a shared Y-axis and two X-axis and the left X-axis will be the count of appliances
    # and the right x-axis will be the component energy consumption
    energy['appliance'] = energy['appliance'].fillna(1)
    prefs = energy[['id', 'prefecture_eng']].drop_duplicates()

    # 1. count the number of appliances
    ckeys = ['id', 'type', 'appliance']
    count = energy[ckeys].pivot_table(index='id', columns='type', values='appliance', aggfunc='count').reset_index()
    count = count.fillna(0)  # replace NAs by 0
    count = count.merge(prefs, on='id', how='left')
    count = count.groupby('prefecture_eng').mean().reset_index()
    # count = count.melt(id_vars=['prefecture_eng'], value_vars=apps)

    # 2. grouped energy consumption
    ckeys = ['id', 'type', 'use']
    use = energy[ckeys].pivot_table(index='id', columns='type', values='use', aggfunc=sum).reset_index()
    use = use.fillna(0)  # replace NAs by 0
    use = use.merge(prefs, on='id', how='left')
    use = use.groupby('prefecture_eng').mean().reset_index()

    # reshape the above data by cities (must do it in two steps because count/sum on the city level does not reflect
    # the average level of ownership or energy consumption
    apps = ['appliance', 'cooking', 'heating', 'vehicle', 'waterheating']
    app_names = ['Appliance', 'Cooking', 'Heater', 'Vehicle', 'Water heater']
    cities = use['prefecture_eng'].values

    # create the canvas
    fig, ax1 = plt.subplots(figsize=(12, 16))

    x_range = range(len(count))  # fixed x-range
    bot_left, bot_right = np.zeros(len(count)), np.zeros(len(count))

    colors = list(WSJ.values())
    for i, a in enumerate(apps):
        # count chart: 0~15
        y_left = -1 * count[a].values
        ax1.barh(x_range, y_left, left=bot_left, alpha=0.65, label=app_names[i], color=colors[i], edgecolor='grey')
        bot_left = bot_left + y_left

    plt.margins(0.01)
    ax2 = ax1.twinx()  # create the new axis
    for i, a in enumerate(apps):
        # use chart: 0~4000
        y_right = use[a].values / 100
        ax2.barh(x_range, y_right, left=bot_right, alpha=0.65, label=app_names[i], color=colors[i], edgecolor='grey')
        bot_right = bot_right + y_right

    # adjust x-axis range
    plt.margins(0.01)
    ax1.set_yticks(x_range, cities, size=12)
    ax1.set_xticks(range(-15, 26, 5), [abs(i) for i in range(-15, 26, 5)], size=14)
    ax1.tick_params(left=False)
    ax1.tick_params(bottom=False)
    ax2.set_yticks([])
    plt.xlim(-15, 25)

    # remove spines for both subplots
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.legend(loc=7, fontsize=14)
    fig.savefig('img/figure2.pdf', format='pdf', dpi=200)
    plt.show()


def city_energy_chart(data: pd.DataFrame) -> pd.DataFrame:
    """
    The dataframe should contain `region` column before analysing and the chart has the following parts:
    - mean values by cities - box plot
    - mean values by urban/rural - box plot
    - mean values by north/south - solid lines

    :param data: a pre-processed dataframe
    """
    data = data[data['en_total'] > 0]  # skip zero-value entries
    out = data[['prefecture_eng']]  # append data to the tot
    # energy consumption by region and city

    data['percap_all'] = data['en_total'] / data['size']
    # in order to correctly rank the cities, we need a customised list of cities
    # the index of Shanghai is 38 (which is the first southern city)
    cities = data.loc[data.region == 0, ['percap_all', 'prefecture_eng']].groupby('prefecture_eng').mean(). \
                 sort_values(['percap_all'], ascending=False).index.tolist() + \
             data.loc[data.region == 1, ['percap_all', 'prefecture_eng']].groupby('prefecture_eng').mean(). \
                 sort_values(['percap_all'], ascending=False).index.tolist()
    # rebuild the chart dataset by the order of cities
    chart = pd.DataFrame()
    for c in cities:
        chart = pd.concat([chart, data[data['prefecture_eng'] == c]], axis=0)
    # ... dataframe for boxplot
    chart['resident'] = chart['resident'].apply(lambda x: 'Urban' if x == 1 else 'Rural')
    # ... dataframe for lineplot
    linechart = data[['region', 'en_total', 'size']].groupby('region').sum().reset_index()
    # north: 432.47804712, south: 255.65224578
    regional = (linechart['en_total'] / linechart['size']).values

    # pre-processing before making the chart
    # result one: percap energy use by urban/rural by cities
    fig = plt.figure(figsize=(10, 16))
    ax = plt.gca()

    sns.boxplot(y='prefecture_eng', x='percap_all', hue='resident', gap=.1,
                data=chart, linewidth=1.5, palette='Set2', fliersize=0)
    plt.xlim(0, 4500)  # the maximum value is <4500
    plt.xticks(size=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.ylabel('Cities', fontsize=12)
    plt.xlabel('Energy consumption per capita (kgce/person)', fontsize=14)
    plt.legend(loc=4, fontsize=12)

    # adjust the margins
    plt.margins(0.01)
    plt.tight_layout()
    fig.savefig('img/figure1.pdf', format='pdf', dpi=200)
    plt.show()


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
    datafile = Path('data') / 'vardata-1030.xlsx'
    data = pd.read_excel(datafile, engine='openpyxl')
    energyfile = Path('data') / 'energyuse-1103.xlsx'
    energy = pd.read_excel(energyfile, engine='openpyxl')
    energy = get_energy(energy)

    merged = data.merge(energy, on='id', how='left')
    # preprocess
    merged = get_cities(merged)
    merged = get_percap(merged, columns=['en_total'])
    _, merged['region'] = create_regional_variable(merged['province'])
    merged.to_excel(Path('data') / 'mergedata-1103.xlsx', index=False)

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

    # energy consumption chart
    city_energy_chart(merged)

    # add a new chart to the result one section
    energy = pd.read_excel(energyfile, engine='openpyxl')
    # use merged data to append the city info [prefecture_eng]
    energy = energy.merge(merged[['id', 'prefecture_eng']], on='id', how='left')
    # reduce 10 categories into 6 categories
    compress_keys = ['ac', 'computer', 'freezing', 'laundry', 'lighting', 'television']
    energy['type'] = energy['type'].apply(lambda x: 'appliance' if x in compress_keys else x)
    city_lifestyle_chart(energy)
