# A quick check over the raw data to find mismatches in the processed data
#

import json
from collections import defaultdict

import pandas as pd
import numpy as np

from pathlib import Path

# config data for mapping
CONFIGFILE = Path('data') / 'config.json'
CONFIG = json.loads(CONFIGFILE.read_text('utf8'))


# aspect-specific calculation
def cooking(row: dict) -> dict:
    """
    Household cooking energy consumption conversion.

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """

    n = row['e13a']

    assert n in CONFIG['cooking'], f'Invalid cooking aspect: {n}'
    conf = CONFIG[n]
    # NB. equation: month-based freq * minute/hour * use_perhour * days-base * kgce

    # 1 * 52.5 / 60 * 0.31 * 3 * 30 * 12 * 1.4129



def checker(proc: pd.DataFrame, raw: pd.DataFrame):
    """

    :param proc: calculated but not verified data
    :param raw: as defined, the unprocessed raw data
    :return: a new dataframe
    """
    sub = proc[proc['id'].isin(raw['id'].values)]  # matched by id
    # missing entries
    intersect = set(raw['id']).intersection(set(raw['id']))
    missing = set(raw['id']).difference(set(raw['id']).intersection(set(proc['id'])))
    print(f'Missing {len(missing)} IDs: {missing}')

    # intersection columns
    overlap = set(sub.columns).intersection(set(raw.columns))
    overlap.remove('id')

    rows = defaultdict(list)
    for i in missing:
        rows['id'] += [i]
        for c in overlap:
            # first: check the intersection
            v = raw.loc[raw['id'] == i, c]
            if not v.empty and v[v.isna()].empty:
                rows[c] += [v.values[0]]
            else:
                rows[c] += ['']

        # second: check the missing

    return


if __name__ == '__main__':
    # load unconverted data
    raw_datafile = Path('data') / 'CGSS-unprocessed-202302.xlsx'
    raw = pd.read_excel(raw_datafile, engine='openpyxl', header=[0], skiprows=[1])

    # load calculated data
    cal_datafile = Path('data') / 'CGSS-calculate-20231019.xlsx'
    sheets = ['烹饪', '冰箱', '洗衣', '电视', '计算机', '照明', '采暖', '热水器', '空调', '交通']  # 总折算

    missings = defaultdict(list)
    for sheet in sheets:
        proc = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name=sheet)
        missings[sheet] = list(set(raw['id']).difference(set(raw['id']).intersection(set(proc['id']))))

    missings = pd.DataFrame(missings)  # raise error if they're in different lengths
    del missings
