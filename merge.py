# Replicable script for the data integration
#
# Created on 19 Oct 2023, by Yi
#

import pandas as pd
from config import VAR_MAPPING

# merging the raw energy use data with the household attribute data
MERGE_MAPPING = {
    '炊具折算': 'en_cooking',
    '冷藏折算': 'en_freezing',
    '洗衣折算': 'en_laundry',
    '电视折算': 'en_television',
    '电脑折算': 'en_computer',
    '灯泡折算': 'en_lighting',
    '采暖折算': 'en_house_heating',
    '热水器折算': 'en_water_heating',
    '空调折算': 'en_ac',
    '汽车折算': 'en_vehicle',
    '总折算': 'en_total'
}


def merging(org: pd.DataFrame, tar: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the original calculated energy use data and the household attribute data.
    NB. if `rawdata-mmdd.csv` is used, please overwrite it and update the data.

    :param org: original data
    :param tar: target data for merging
    :return: a new dataframe
    """
    # reshape the org data (3863 -> 3648)
    org.index = org['id']
    org = org.drop(columns=['id'])
    org = org.dropna(how='all')  # drop NAs (incl. residents without energy data)
    org = org.reset_index()
    # rename by the mapping
    names = [MERGE_MAPPING.get(c, 'id') for c in org.columns]
    org.columns = names

    # report the diff
    diff = set(var['id']).difference(set(org.index))  # find the diff household IDs
    print(f'Difference: {len(var) - len(org)} in [{diff}]')
    # merging
    return tar.merge(org, on='id', how='left')


if __name__ == '__main__':
    from pathlib import Path
    import numpy as np

    # load calculated data
    cal_datafile = Path('data') / 'CGSS-calculate-20231019.xlsx'
    calculate = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name='总折算')

    # load the latest variable data
    tag = '0229'
    # NB. changed on 2024-02-25, was vardata-0207.xlsx.
    var_datafile = Path('data') / f'vardata-{tag}.xlsx'
    var = pd.read_excel(var_datafile, engine='openpyxl')
    # remove all the emission columns
    drops = [it for it in var.columns if 'emi' in it]
    var = var.drop(columns=drops)

    merged = merging(calculate, var)
    merge_datafile = Path('data') / f'mergedata-{tag}.xlsx'
    merged.to_excel(merge_datafile, index=False)

    # output descriptive summary
    datafile = Path('data') / 'mergedata-0229.xlsx'
    raw = pd.read_excel(datafile, engine='openpyxl')
    raw = raw.drop(columns=['media', 'aware_trust', 'aware_harm',
                            'aware_justice', 'aware_happy', 'job', 'fuel_vehicle'])

    # preprocessing and criteria filtering
    data = raw[(raw['en_total'] > 0 & ~raw['region'].isna())]
    na_safe_keys = ['children_num', 'elderly_num', 'num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
                    'num_water_heater', 'freq_water_heater', 'time_water_heater', 'num_ac', 'freq_ac', 'power_ac',
                    'time_ac', 'label_water_heater', 'label_ac', 'type_heating', 'time_heating', 'area_heating',
                    'cost_heating', 'own_vehicle', 'fuel_price_vehicle', 'cost_vehicle', 'vehicle_dist',
                    'vehicle_use', 'vehicle_fuel', 'raw_income', 'expenditure', 'income_percap']
    data[na_safe_keys] = data[na_safe_keys].replace(-99, np.nan)
    data['en_total_percap'] = data['en_total'].values / data['size'].values
    data['province_id'] = data['province'].apply(lambda x: list(data['province'].unique()).index(x))
    data['vehicle_dist'] = data['vehicle_dist'].replace(0, np.nan)
    data.loc[data['vehicle_dist'].isna(), ['vehicle_fuel', 'vehicle_use']] = np.nan

    # keys in the manuscript
    keys = ['prefecture_id', 'region', 'age', 'house_area', 'size', 'children_num', 'elderly_num',
            'expenditure', 'raw_income', 'outside', 'live_days',
            'if_single_elderly', 'if_singleAE', 'if_singleA', 'if_couple_elderly', 'if_coupleA',
            'if_singleWithChildren', 'if_coupleWithChildren', 'if_grandparentKids', 'if_bigFamily', 'if_existElderly',
            'num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
            'num_water_heater', 'freq_water_heater', 'time_water_heater', 'label_water_heater',
            'num_ac', 'freq_ac', 'power_ac', 'time_ac', 'label_ac',
            'type_heating', 'time_heating', 'area_heating', 'cost_heating',
            'vehicle_num', 'fuel_price_vehicle', 'cost_vehicle', 'vehicle_dist', 'vehicle_fuel', 'vehicle_use',
            'en_total', 'en_total_percap']
    names = [VAR_MAPPING[k] for k in keys]
    summary = data[keys].describe().reset_index().T
    summary.index = ['stats'] + names
    summary.to_excel(Path('data') / 'clustered-0223' / 'descriptive-summary.xlsx', index=True)
