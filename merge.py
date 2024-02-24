# Replicable script for the data integration
#
# Created on 19 Oct 2023, by Yi
#

import pandas as pd

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
    diff = set(var['id']).difference(set(org['id']))  # find the diff household IDs
    org.id = org['id']
    org = org.drop(columns=['id'])
    org = org.dropna(how='all')  # drop NAs (incl. residents without energy data)
    # rename by the mapping
    names = [MERGE_MAPPING[c] for c in org.columns]
    org = org.reset_index()
    org.columns = ['id'] + names
    # report the diff
    diff = set(var['id']).difference(set(org.index))  # find the diff household IDs
    print(f'Difference: {len(var) - len(org)} in [{diff}]')
    # merging
    return tar.merge(org, on='id', how='left')


if __name__ == '__main__':
    from pathlib import Path

    # load calculated data
    cal_datafile = Path('data') / 'CGSS-calculate-20231019.xlsx'
    calculate = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name='总折算')

    # load the latest variable data
    var_datafile = Path('data') / 'vardata-0207.xlsx'
    var = pd.read_excel(var_datafile, engine='openpyxl')
    # remove all the emission columns
    drops = [it for it in var.columns if 'emi' in it]
    var = var.drop(columns=drops)

    merged = merging(calculate, var)

