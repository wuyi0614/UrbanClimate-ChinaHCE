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


# redo the data merging
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


def add_general(org: pd.DataFrame, tar: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """
    Add up general data to the original dataframe

    :param org: original dataframe
    :param tar: target dataframe
    :return: an updated dataframe
    """
    sub = tar[['id'] + columns]
    return org.merge(sub, on='id', how='left')


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

        # calculate, and the conversion equation between stove heat load and kgce
        # assume heat load=2kw （~7.2MJ/h eq), then use_perhour = heat load / fuel's heat (MJ)
        # e.g. civil briquette's heat is 23MJ, then its perhour use is 7.2MJ/h / 23MJ/kg = 0.313 kg/h
        ff = freq.get(f, 0) * 12
        tt = time.get(t, 0) / 60

        if '灶' in n or '蜂窝' in n:
            r = 12 * ff * tt * c['use_perhour'] * c['coal_base']
        else:
            use_perhour = power.get(p, 0) / 1000
            r = ff * tt * use_perhour * c['coal_base']

        # save results
        result['id'] += [row['id']]
        result['type'] += ['cooking']
        result['appliance'] += [n]
        result['power'] += [p]
        result['efficiency'] += ['']
        result['frequency'] += [ff]
        result['time'] += [tt]
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
        tt = time.get(t, 0) / 60
        ff = freq.get(f, 0) * 12
        # power/hour * hour * times/month * months * coal_base
        r = pp * tt * ff * coal_base

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
        tt = time.get(t, 0) / 60
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * 30 * 12 * coal_base

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
    Household computer energy consumption conversion.
    Mapping (e31x-e5x):
        - e31a, type
        - e31e, monitor efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e31b, monitor size
        - e31f, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['computer']
    for i, idx in enumerate(range(31, 36)):
        n = str(row[f'e{idx}a']).strip()
        if n == 'nan' or n == '':
            continue

        # load config
        time = CONFIG['time']

        # extract the parameters
        e = str(row[f'e{idx}e']).strip()  # efficiency
        s = str(row[f'e{idx}b']).strip()  # monitor size
        t = str(row[f'e{idx}f']).strip()  # time

        # check the answer
        c = conf.get(n, {})
        m = conf.get(s, {})
        p = c.get('mean_power', 0)  # power
        pm = m.get('mean_power', 0)  # monitor's power
        pp = p + pm if n == '台式机' else p
        ss = m.get('mean_size', 0)  # monitor's size

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0) / 60
        pp = pp / 1000
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['computer']
        result['appliance'] += [f'{n}[{ss}-inch]']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def lighting(row: dict) -> dict:
    """
    Household lighting energy consumption conversion.
    Mapping (e36x-e38x):
        - e36a, number of bulbs
        - e36b, use time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['lighting']
    for i, idx in enumerate(range(36, 39)):
        # NB. equation:
        time, number = CONFIG['time'], CONFIG['number']
        # extract the parameters
        n = str(row[f'e{idx}a']).strip()  # number
        t = str(row[f'e{idx}b']).strip()  # time
        # check the answer
        if n == 'nan' or n == '':
            continue

        pp = list(conf.values())[i]
        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0) / 60
        nn = number.get(n, 1)

        # power/hour * use hour * number * runtime * days * coal_base
        r = pp/1000 * nn * tt * 30 * 12 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['lighting']
        result['appliance'] += [list(conf.keys())[i]+f'[{nn}bulbs]']
        result['power'] += [pp]
        result['efficiency'] += ['']
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def heating(row: dict) -> dict:
    """
    Household heating energy consumption conversion.
    Mapping (e52x-e55x):
        - e52a, name
        - e52b, fuel type
        - e52c, time
        - e52d, freq
        - e52f, area, m2

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
        tt = time.get(t, 0) / 60
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * 30 * 12 * coal_base

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


def add_heating(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping:
        - e39, heating source, 1-centric, 2-individual, 3-mixed, 4-no
        - e41, heating mode, 1-steam, 2-hot water, 3-hot wind
        - e42, run time, months
        - e44, heating area
        - e46, temperature

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
        tt = time.get(t, 0) / 60
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * 30 * 12 * coal_base

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


def waterheating(row: dict) -> dict:
    """
    Household water heating energy consumption conversion.
    Mapping (e57x-e58x):
        - e57a, name/type
        - e57b, fuel type
        - e57e, frequency
        - e57f, time
        - e57g, efficiency

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    conf = CONFIG['waterheating']
    time, freq, fuel = CONFIG['time'], CONFIG['freq'], CONFIG['fuel']
    eff = CONFIG['ee']['waterheating']
    for i, idx in enumerate(range(57, 59)):
        # extract the parameters
        n = str(row[f'e{idx}a']).strip()  # name/type
        y = str(row[f'e{idx}b']).strip()  # fuel
        e = str(row[f'e{idx}g']).strip()  # efficiency
        t = str(row[f'e{idx}f']).strip()  # time
        f = str(row[f'e{idx}e']).strip()  # freq
        if n == 'nan' or n == '':
            continue

        # check the answer
        use_perhour = conf[n][y]['mean_power']  # which type with what fuel
        ff = freq.get(f, 0) / 30
        tt = time.get(t, 0) / 60
        ee = eff.get(e, 0)

        # equation:
        # 1. storage water heater
        #    power(kW) * work hour(hour/times)[regular + use hours(0.5*times/th)] * eff * (days/year) * coal base
        # 2. instant water heater
        if n == '储水式':
            coal_base = 0.1229
            work_hour_regular = 3  # 3 hours per day if it runs for 24h
            threshold_work = 1.025  # actual freq = frequency / threshold_work (average use times per family)
            actual_freq = ff / threshold_work
            r = use_perhour * (work_hour_regular + actual_freq * 0.5) * ee * 365 * coal_base
        else:
            if y == '电力':
                coal_base = 0.1229
                r = use_perhour * tt * 365 * coal_base
            elif '太阳能' in y:
                r = use_perhour * 365
            else:
                r = use_perhour * tt * 365 * fuel[y]

        # save results
        result['id'] += [row['id']]
        result['type'] += ['waterheating']
        result['appliance'] += [f'{n}[{y}]']
        result['power'] += [use_perhour]
        result['efficiency'] += [e]
        result['frequency'] += [ff]
        result['time'] += [tt]
        result['use'] += [r]

    return result


def ac(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e60x-e63x):
        - e60d, name/type
        - e60e, energy efficiency
        - e60f, fixed/flexible
        - e60g, frequency
        - e60h, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    time, power, runtime = CONFIG['time'], CONFIG['ac'], CONFIG['runtime']
    eff = CONFIG['ee']['ac']
    result = defaultdict(list)
    for i, idx in enumerate(range(60, 64)):
        # extract the parameters
        p = str(row[f'e{idx}d']).strip()  # power
        e = str(row[f'e{idx}e']).strip()  # efficiency
        a = str(row[f'e{idx}f']).strip()  # fixed/flexible
        f = str(row[f'e{idx}g']).strip()  # frequency
        t = str(row[f'e{idx}h']).strip()  # time

        # check the answer
        pp = power.get(p, 0)
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate for electricity
        adj = 0.7 if '变频' in a else 1
        ee = eff.get(e, 0)
        tt = time.get(t, 0)
        rt = runtime.get(f, 0)

        # equation:
        # output power (kW) * adjust / efficiency * time (hour/day) * run time(days) * coal base
        r = pp * adj / ee * tt * rt * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['ac']
        result['appliance'] += [f'ac[{pp}kW]']
        result['power'] += [pp]
        result['efficiency'] += [e]
        result['frequency'] += [rt]
        result['time'] += [tt]
        result['use'] += [r]

    return result


def vehicle(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e67, e70, e68, e72):
        - e67, engine fuel use
        - e68, driving distance
        - e70, fuel type
        - e72, actual fuel use

    Notes: 1. mixed fuel is assumed to be used in half-half
           2.

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    result = defaultdict(list)
    dist, fuel, use = CONFIG['distance'], CONFIG['fuel'], CONFIG['vehicle']

    # parameters from conversion
    # en = str(row[f'e67']).strip()  # engine fuel use, ?L
    di = str(row[f'e68']).strip()  # distance, 10,000km
    fu = str(row[f'e70']).strip()  # fuel type
    us = str(row[f'e72']).strip()  # actual fuel use, ?L/100km

    # calculation has two parts:
    # fossil fuel vehicles:
    # energy(kgce/year) = actual fuel use(L/100km) * distance(100km/year) * coal base
    r = use.get(us, 0) * dist.get(di, 0) * 100 * fuel.get(fu)

    # save results
    result['id'] += [row['id']]
    result['type'] += ['vehicle']
    result['appliance'] += ['car']
    result['power'] += ['']
    result['efficiency'] += ['']
    result['frequency'] += ['annual']
    result['time'] += ['']
    result['use'] += [r]
    return result


def add_vehicle(row: dict) -> dict:
    """
    Add vehicle-related variables to our dataset
    Mapping:
        - e39, heating source, 1-centric, 2-individual, 3-mixed, 4-no
        - e41, heating mode, 1-steam, 2-hot water, 3-hot wind
        - e42, run time, months
        - e44, heating area
        - e46, temperature

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
        tt = time.get(t, 0) / 60
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * 30 * 12 * coal_base

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


def add_fuel(row: dict) -> dict:
    """
    Add fuel-related variables to our dataset.
    Mapping (1-10):
        e89, fuel type used in households
             蜂窝煤/煤球,
             煤块,
             汽油,
             柴油,
             瓶装液化气,
             管道天然气,
             管道煤气,
             畜禽粪便,
             秸秆,
             薪柴

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """

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

    # check missing data
    proc = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name='空调')
    missing = checker(proc, raw, mode='all')

    # unittest for each source of energy use
    row = proc.loc[2, :].to_dict()
    cooking(row)
