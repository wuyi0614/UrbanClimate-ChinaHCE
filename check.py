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
            r = 12 * freq.get(f, 0) * time.get(t, 0) / 60 * c['use_perhour'] * c['coal_base']
        else:
            use_perhour = power.get(p, 0) / 1000
            r = 12 * freq.get(f, 0) * time.get(t, 0) / 60 * use_perhour * c['coal_base']

        # save results
        result['id'] += [row['id']]
        result['type'] += ['cooking']
        result['appliance'] += [n]
        result['power'] += [p]
        result['efficiency'] += ['']
        result['frequency'] += [f]
        result['time'] += [t]
        result['use'] += [r]

    return result


def freezing(row: dict) -> dict:
    """
    Household freezing energy consumption conversion.
    Mapping (e18x-e20x):
        - e18a, size
        - e18d, power
        - e18e, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e18f, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['freezing']
    for i, idx in enumerate(range(18, 21)):
        # use name of appliance for verification
        n = str(row[f'e{idx}a']).strip()
        if n == 'nan':
            continue

        assert n in conf, f'Invalid freezing appliance: {n}'
        c = conf[n]
        # NB. equation:
        runtime, power, eff = CONFIG['runtime'], CONFIG['power'], CONFIG['ee']['freezing']

        # extract the parameters
        p = str(row[f'e{idx}d']).strip()  # power
        e = str(row[f'e{idx}e']).strip()  # ee
        t = str(row[f'e{idx}f']).strip()  # runtime

        # calculate
        coal_base = 0.1229  # fixed rate
        pp = c.get('mean_power', 0)
        s = c.get('mean_size', 0)
        pp = power.get(p, 0) if power.get(p, 0) > 0 else pp
        ee = eff.get(e, 0)  # no label = 1
        rt = runtime.get(t, 0)
        # power/hour * hour * efficiency rate * runtime * days * coal_base
        r = pp * 24 * ee * rt * 30 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['freezing']
        result['appliance'] += [f'fridge[{s}L]']
        result['power'] += [pp]
        result['efficiency'] += [ee]
        result['frequency'] += ['']
        result['time'] += [rt]
        result['use'] += [r]

    return result


def laundry(row: dict) -> dict:
    """
    Household laundry energy consumption conversion.
    Mapping (e22x-e24x):
        - e22a, size
        - e22d, power
        - e22e, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e22f, freq
        - e22g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['laundry']
    for i, idx in enumerate(range(22, 25)):
        # use name of appliance for verification
        n = str(row[f'e{idx}a']).strip()
        if n == 'nan':
            continue

        assert n in conf, f'Invalid laundry appliance: {n}'
        c = conf[n]
        time, power, freq = CONFIG['time'], CONFIG['power'], CONFIG['freq']
        eff = CONFIG['ee']['laundry']

        # extract the parameters
        p = str(row[f'e{idx}d']).strip()  # power
        e = str(row[f'e{idx}e']).strip()  # efficiency
        f = str(row[f'e{idx}f']).strip()  # freq
        t = str(row[f'e{idx}g']).strip()  # time

        # calculate the electricity consumption
        coal_base = 0.1229  # fixed rate
        ee = eff.get(e, 0)  # no label = 1
        # get mean power
        s = c['mean_size']
        mean_power = s * c['mean_cycle'] * ee  # mean power kwh
        pp = power[p] if power[p] > 0 else mean_power
        # get time and freq
        tt = time.get(t, 0)
        ff = freq.get(f, 0)
        # power/hour * hour * times/month * months * coal_base
        r = pp * tt / 60 * ff * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['laundry']
        result['appliance'] += [f'laundry[{s}kg]']
        result['power'] += [pp]
        result['efficiency'] += [ee]
        result['frequency'] += [ff]
        result['time'] += [tt]
        result['use'] += [r]

    return result


def television(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # NB. equation:
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def computer(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e31x-e36x):
        - e31a, type
        - e31e, monitor efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e31f, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        n = str(row[f'e{idx}a']).strip()
        if n == 'nan':
            continue

        # load config
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def lighting(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # NB. equation:
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def heater(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # NB. equation:
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def ac(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # NB. equation:
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def vehicle(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # NB. equation:
        time, power = CONFIG['time'], CONFIG['power']

        # extract the parameters
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0)
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt/60 * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += ['']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def checker(proc: pd.DataFrame, raw: pd.DataFrame, mode='find'):
    """

    :param proc: calculated but not verified data
    :param raw: as defined, the unprocessed raw data
    :param mode: find=missing ids only, all=all ids
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

    # depend on the mode: find / all
    ids = missing if mode == 'find' else raw['id'].unique()
    rows = defaultdict(list)
    for i in ids:
        rows['id'] += [i]
        for c in overlap:
            # first: check the intersection
            v = raw.loc[raw['id'] == i, c]
            if not v.empty and v[v.isna()].empty:
                rows[c] += [v.values[0]]
            else:
                rows[c] += ['']

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
    proc = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name='电视')
    missing = checker(proc, raw)

    # calculate energy use
    row = proc.loc[2, :].to_dict()
    cooking(row)
