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
    Mapping (e13x-e16x):
        - e13a, name
        - e13c, power
        - e13d, freq
        - e13e, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['cooking']
    for i, idx in enumerate(range(13, 17)):
        # use name of appliance for verification
        n = str(row[f'e{idx}a']).strip()
        if n == 'nan':
            print('No appliance specified')
            continue

        assert n in conf, f'Invalid cooking appliance: {n}'
        c = conf[n]
        # NB. equation:
        # months * month-based freq * minute/hour * use_perhour * kgce
        freq, time, power = CONFIG['freq'], CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}c']).strip()  # watts
        f = str(row[f'e{idx}d']).strip()  # days
        t = str(row[f'e{idx}e']).strip()  # minutes

        # calculate
        if n.startswith('煤气灶'):
            r = 12 * freq.get(f, 0) * time.get(t, 0)/60 * c['use_perhour'] * c['coal_base']
        else:
            use_perhour = p/1000
            r = 12 * freq.get(f, 0) * time.get(t, 0)/60 * use_perhour * c['coal_base']

        # save results
        result['id'] += [row['id']]
        result['type'] += ['cooking']
        result['appliance'] += [n]
        result['power'] += [p]
        result['frequency'] += [f]
        result['time'] += [t]
        result['use'] += [r]

    return result


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

    return pd.DataFrame(rows)


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

    # check missing data
    missing = checker(proc, raw)

