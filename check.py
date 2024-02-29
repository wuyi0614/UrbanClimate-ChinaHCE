# A quick check over the raw data to find mismatches in the processed data
#

import json
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from pathlib import Path

# config data for mapping
CONFIGFILE = Path('data') / 'config.json'
CONFIG = json.loads(CONFIGFILE.read_text('utf8'))

MAPPINGFILE = Path('data') / 'mapping.json'
MAPPING = json.loads(MAPPINGFILE.read_text('utf8'))


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


def add_general(org: pd.DataFrame, tar: pd.DataFrame, columns: list) -> pd.DataFrame:
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
        - e2, days in house (days/week)
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
        w = str(row[f'e2']).strip()  # days/week
        p = str(row[f'e{idx}c']).strip()  # watts
        f = str(row[f'e{idx}d']).strip()  # days
        t = str(row[f'e{idx}e']).strip()  # minutes

        # calculate, and the conversion equation between stove heat load and kgce
        # assume heat load=2kw （~7.2MJ/h eq), then use_perhour = heat load / fuel's heat (MJ)
        # e.g. civil briquette's heat is 23MJ, then its perhour use is 7.2MJ/h / 23MJ/kg = 0.313 kg/h
        ww = float(w) * 52 if w else 0  # assume 52 weeks a year
        ff = freq.get(f, 0) / 30
        tt = time.get(t, 0) / 60

        if '灶' in n or '蜂窝' in n:
            r = ww * ff * tt * c['use_perhour'] * c['coal_base']
        else:
            use_perhour = power.get(p, 0) / 1000
            r = ww * ff * tt * use_perhour * c['coal_base']

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
        e = str(row[f'e{idx}e']).strip()  # ee
        t = str(row[f'e{idx}f']).strip()  # runtime

        # calculate
        coal_base = 0.1229  # fixed rate
        s = c.get('mean_size', 0)
        ee = eff.get(e, 0)  # no label = 1
        rt = runtime.get(t, 0)

        # yeta * (M * V + n)/365 * days in house
        # M = kWh/L, 0.526 (fridge), 0.567 (freezer)
        # N = kWh, 228 (fridge), 205 (freezer)
        if idx < 21:  # fridges
            M, N = 0.526, 228
        else:
            M, N = 0.567, 205

        p = ee * (M * s + N) / 365
        r = p * rt * 30 * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['freezing']
        result['appliance'] += [f'fridge[{s}L]']
        result['power'] += [p]
        result['efficiency'] += [ee]
        result['frequency'] += ['']
        result['time'] += [rt]
        result['use'] += [r]

    return result


def laundry(row: dict) -> dict:
    """
    Household laundry energy consumption conversion.
    Mapping (e22x-e24x):
        - e2, days in house
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
    w = str(row['e2']).strip()  # days/week in house
    ww = float(w) * 52 if w else 0
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
        # get time and freq
        tt = time.get(t, 0) / 60
        ff = freq.get(f, 0) / 30 * ww
        # get mean power
        s = c['mean_size']
        if power[p] > 0:
            # power/hour * hour * times/month * months * coal_base
            pp = power[p] / 1000
            r = pp * tt * ff * coal_base
        else:
            # size * electricity use per circle per kg * ee * frequency
            # 0.3 kWh/kg*circle (circle = use time / 45 (average use time))
            r = s * tt * 60 / 45 * ee * ff * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['laundry']
        result['appliance'] += [f'laundry[{s}kg]']
        result['power'] += ['']
        result['efficiency'] += [ee]
        result['frequency'] += [ff]
        result['time'] += [tt]
        result['use'] += [r]

    return result


def television(row: dict) -> dict:
    """
    Household television energy consumption conversion.
    Mapping (e26x-e28x):
        - e2, days in house
        - e26e, power
        - e26f, efficiency, 1-no, 2-1st EE, 6-5th EE.
        - e26g, time

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    time, power = CONFIG['time'], CONFIG['power']
    result = defaultdict(list)
    for i, idx in enumerate(range(26, 29)):
        # extract the parameters
        w = str(row[f'e2']).strip()  # days/week in house
        p = str(row[f'e{idx}e']).strip()  # power
        e = str(row[f'e{idx}f']).strip()  # efficiency
        t = str(row[f'e{idx}g']).strip()  # time
        # check the answer
        pp = power.get(p, 0) / 1000
        if pp == 0:
            continue

        # calculate
        coal_base = 0.1229  # fixed rate
        ww = float(w) * 52 if w else 0  # days/year
        tt = time.get(t, 0) / 60
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * ww * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['television']
        result['appliance'] += [f'TV{i}']
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
        - e2, days/week in house
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
        w = str(row[f'e2']).strip()  # days/week in house
        e = str(row[f'e{idx}e']).strip()  # efficiency
        s = str(row[f'e{idx}b']).strip()  # monitor size
        t = str(row[f'e{idx}f']).strip()  # time

        # check the answer
        c = conf.get(n, {})
        m = conf.get(s, {})
        ww = float(w) * 52 if w else 0
        p = c.get('mean_power', 0)  # power
        pm = m.get('mean_power', 0)  # monitor's power
        pp = p + pm if n == '台式机' else p
        ss = m.get('mean_size', 0)  # monitor's size

        # calculate
        coal_base = 0.1229  # fixed rate
        tt = time.get(t, 0) / 60
        pp = pp / 1000
        # power/hour * use hour * runtime * days * coal_base
        r = pp * tt * ww * coal_base

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
        - e2, days/week in house
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
        w = str(row[f'e2']).strip()  # days/week in house
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
        ww = float(w) * 52 if w else 0

        # power/hour * use hour * number * runtime * days * coal_base
        r = pp / 1000 * nn * tt * ww * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['lighting']
        result['appliance'] += [list(conf.keys())[i] + f'[{nn}bulbs]']
        result['power'] += [pp]
        result['efficiency'] += ['']
        result['frequency'] += ['everyday']
        result['time'] += [tt]
        result['use'] += [r]

    return result


def _centric_heating(row: dict) -> dict:
    runtime, area, fuel, year, heat = CONFIG['runtime'], CONFIG['area'], CONFIG['fuel'], CONFIG['year'], CONFIG[
        'heating']

    t = str(row['e39']).strip()  # heating type
    r = str(row['e42']).strip()  # run time
    a = str(row['e44']).strip()  # area
    y = str(row['e3']).strip()   # construction year
    m1 = str(row['e7']).strip()  # window/door, -10%
    m2 = str(row['e8']).strip()  # wall, -30%
    m3 = str(row['e9']).strip()  # ceiling, -10%

    # converted the above
    yy = heat.get(y, 0)  # ? kgce/m2, energy use / unit area
    m1 = year.get(m1, 0) * 0.1
    m2 = year.get(m2, 0) * 0.3
    m3 = year.get(m3, 0) * 0.1

    result = defaultdict(list)
    tt = runtime.get(r, 3.43) / 3.43  # heating run time (months)
    aa = area.get(a, 115)  # 115 m2 as default for the average (m2)
    r = yy * (1 - m1) * (1 - m2) * (1 - m3) * tt * aa

    result['id'] += [row['id']]
    result['type'] += ['heating']
    result['appliance'] += ['centric heating']
    result['power'] += [yy]
    result['efficiency'] += ['']
    result['frequency'] += ['']
    result['time'] += [tt]
    result['use'] += [r]
    return result


def _self_heating(row) -> dict:
    area, fuel, time, heat = CONFIG['area'], CONFIG['fuel'], CONFIG['time'], CONFIG['heating']
    runtime = CONFIG['runtime']
    result = defaultdict(list)
    for i, idx in enumerate(range(53, 56)):
        iac = 0
        # extract the parameters
        y = str(row[f'e{idx}a']).strip()  # heater type
        ft = str(row[f'e{idx}b']).strip()  # fuel type
        t = str(row[f'e{idx}d']).strip()  # use time
        rt = str(row[f'e{idx}c']).strip()  # run time
        a = str(row[f'e{idx}f']).strip()  # area, m2

        tt = time.get(t, 0) / 60  # hours
        rt = runtime.get(rt, 0) * 30  # months
        aa = area.get(a, 0)  # area, m2

        if '空调' in y:
            coal_base = 0.1229  # electricity
            rac = ac(row)  # extract power, ee
            if rac['use'] and iac <= len(rac['use']):
                adj = 0.7 if '变频' in rac['appliance'] else 1
                r = rac['power'][iac] * adj / rac['efficiency'][iac] * tt * rt * coal_base
                iac += 1  # count + 1 if the order of ACs is not consistent
            else:
                r = 0
        elif '地膜' in y or '电辐射' in y:
            coal_base = 0.1229  # electricity
            p = 1.2  # power, kW as default
            r = p * tt * rt * coal_base
        else:
            # unit area of heat load * area * hours * days * coal base
            # convert fuel types
            if ft == 'ú':
                ft = '木炭'
            elif ft == 'ľ̿':
                ft = '煤'

            ff = heat.get(ft, 0)
            coal_base = fuel.get(ft, 0)
            if ft == '瓶装液化气':
                ff = ff * 0.55  # m3 -> kg

            r = ff * aa * tt * rt * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['heating']
        result['appliance'] += [f'{y}[{ft}]']
        result['power'] += ['']
        result['efficiency'] += ['']
        result['frequency'] += [rt]
        result['time'] += [tt]
        result['use'] += [r]

    return result


def heating(row: dict) -> dict:
    """
    Household heating energy consumption conversion.
    Mapping (e53x-e55x + e60x-e63x):
        - e3, construction year
        - e7, door+window sealing
        - e8, outer wall warming
        - e9, cabin/ceiling warming
        - e39, heating type
        - e42, time for centric heating
        - e44, area for centric heating
        - e53a, name
        - e53b, fuel type
        - e53c, time
        - e53d, run time
        - e53f, area, m2
        - variables for AC

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    # load configs
    t = str(row['e39']).strip()  # heating type
    result = defaultdict(list)
    if t == '集中式供暖':
        return _centric_heating(row)
    elif t == '分户自供暖':
        return _self_heating(row)
    elif t == '没有供暖':
        r = 0
    else:  # mixed (0.5 * centric + 0.5 * self)
        r1 = _centric_heating(row)
        r2 = _self_heating(row)
        r = (sum(r1['use']) + sum(r2['use'])) / 2

    # fill up data and return values
    result['id'] += [row['id']]
    result['type'] += ['heating']
    result['appliance'] += [t]
    result['power'] += ['']
    result['efficiency'] += ['']
    result['frequency'] += ['']
    result['time'] += ['']
    result['use'] += [r]
    return result


def waterheating(row: dict) -> dict:
    """
    Household water heating energy consumption conversion.
    Mapping (e57x-e58x):
        - e2, days/week in house
        - e57a, name/type
        - e57b, fuel type
        - e57e, frequency
        - e57f, time
        - e57g, efficiency

    :param row: a dict(row-like) data from dataframe
    :return: an updated dict
    """
    conf = CONFIG['waterheating']
    time, freq, fuel = CONFIG['time'], CONFIG['freq'], CONFIG['fuel']
    eff = CONFIG['ee']['waterheating']

    w = str(row['e2']).strip()
    ww = float(w) * 52 if w else 0

    result = defaultdict(list)
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
        ff = freq.get(f, 0) / 30
        tt = time.get(t, 0) / 60
        ee = eff.get(e, 0)

        # equation:
        # 1. storage water heater
        #    power(kW) * work hour(hour/times)[regular + use hours(0.5*times/th)] * eff * (days/year) * coal base
        # 2. instant water heater
        if n == '储水式':
            use_perhour = 1.5 if y == '太阳能' else 2  # storage water heater only has solar/electric types
            coal_base = 0.1229
            work_hour_regular = 3  # 3 hours per day if it runs for 24h
            threshold_work = 1.025  # actual freq = frequency / threshold_work (average use times per family)
            actual_freq = ff / threshold_work
            r = use_perhour * (work_hour_regular + actual_freq * 0.5) * ee * ww * coal_base
        else:
            use_perhour = conf[n][y]['mean_power']  # which type with what fuel
            if y == '电力':
                coal_base = 0.1229
                r = use_perhour * tt * ww * coal_base
            elif '太阳能' in y:
                r = use_perhour * ww
            else:
                r = use_perhour * tt * ww * fuel[y]

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
        tt = time.get(t, 0) / 60
        rt = runtime.get(f, 0) * 30  # raw unit: month

        # equation:
        # output power (kW) * adjust / efficiency * time (hour/day) * run time(days) * coal base
        r = pp * adj / ee * tt * rt * coal_base

        # save results
        result['id'] += [row['id']]
        result['type'] += ['ac']
        result['appliance'] += [f'ac[{pp}kW{a}]']
        result['power'] += [pp]
        result['efficiency'] += [ee]
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

    Notes: mixed fuel is assumed to be used in half-half

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
    r = use.get(us, 0) * dist.get(di, 0) * 100 * fuel.get(fu, 0)

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


def add_vehicle(org: pd.DataFrame, raw:pd.DataFrame) -> pd.DataFrame:
    """
    Add vehicle-related variables to our dataset
    Mapping:
        - e64, if having cars or not
        - e68, driving distance
        - e70, fuel type
        - e72, cleaner vehicles

    :param org: the original dataframe that needs merging on
    :param raw: the raw dataframe where the variables come from
    :return: an updated dataframe
    """
    # mapping configuration
    dist, use = CONFIG['distance'], CONFIG['vehicle']
    fuel = MAPPING['fuel']['vehicle']

    result = defaultdict(list)
    for _, row in tqdm(raw.iterrows(), desc='Adding vehicle:'):
        di = str(row['e64']).strip()  # 1/0, having cars
        ds = str(row['e68']).strip()  # driving distance
        fu = str(row[f'e70']).strip()  # fuel type
        us = str(row[f'e72']).strip()  # actual fuel use, ?L/100km

        di = 1 if '是' in di else 0
        ds = dist.get(ds, 0)
        fu = fuel.get(fu, 7)  # 7 is the other/unrecognised
        us = use.get(us, 0)

        result['id'] += [row['id']]
        result['vehicle_num'] += [di]
        result['vehicle_dist'] += [ds]
        result['vehicle_fuel'] += [fu]
        result['vehicle_use'] += [us]

    # merging up
    result = pd.DataFrame(result)
    return org.merge(result, on='id', how='left')


def add_fuel(org: pd.DataFrame, raw:pd.DataFrame) -> pd.DataFrame:
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

    :param org: the original dataframe that needs merging on
    :param raw: the raw dataframe where the variables come from
    :return: an updated dataframe
    """
    result = defaultdict(list)
    for _, row in tqdm(raw.iterrows(), desc='Adding fuel:'):
        result['id'] += [row['id']]
        for i in range(1, 11):
            fu = str(row[f'e89_{i}']).strip()  # fuel type
            result[f'fuel{i}'] += [1 if '是' in fu else 0]

    result = pd.DataFrame(result)
    return org.merge(result, on='id', how='left')


def main(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all energy consumption for each household

    :param data: the dataframe where each line represents a family (record)
    :return: a dataframe with `result` structure
    """
    funcs = [cooking, freezing, heating, laundry, television,
             computer, lighting, waterheating, ac, vehicle]

    df = pd.DataFrame()
    for _, row in tqdm(data.iterrows(), desc='Calculating energy use:'):
        row = row.to_dict()
        for f in funcs:
            r = pd.DataFrame(f(row))
            df = pd.concat([df, r], axis=0)

    return df


if __name__ == '__main__':
    # load unconverted data
    raw_datafile = Path('data') / 'CGSS-updateid-202402.xlsx'
    raw = pd.read_excel(raw_datafile, engine='openpyxl', header=[0], skiprows=[1])

    # load calculated data
    tag = '0229'
    cal_datafile = Path('data') / 'CGSS-calculate-20231019.xlsx'
    # sheets = ['烹饪', '冰箱', '洗衣', '电视', '计算机', '照明', '采暖', '热水器', '空调', '交通']  # 总折算
    # check missing data
    proc = pd.read_excel(cal_datafile, engine='openpyxl', sheet_name='空调')
    missing = checker(proc, raw, mode='all')
    missing = add_general(missing, raw, columns=['e2'])

    # add up variables
    varfile = Path('data') / f'vardata-{tag}.xlsx'
    var = pd.read_excel(varfile, engine='openpyxl')
    var = add_vehicle(var, raw)
    var = add_fuel(var, raw)
    var.to_excel(Path('data') / f'vardata-{tag}.xlsx', index=False)

    # unittest for each source of energy use
    for i in range(0, 10):
        row = raw.loc[i, :].to_dict()
        cooking(row)
        freezing(row)
        heating(row)
        laundry(row)
        television(row)
        computer(row)
        lighting(row)
        waterheating(row)
        ac(row)
        vehicle(row)

    # overall test
    use = main(raw)
    use.to_excel(Path('data') / f'energyuse-{tag}.xlsx', index=False)
