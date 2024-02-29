# The script for typology identification
#
import pandas as pd


def create_matrix(d: pd.DataFrame, cluster_key: str):
    keys = {'general': ['region', 'prefecture_id', 'size', 'age', 'outside', 'live_days', 'house_area'],
            'economic': ['raw_income', 'expenditure', 'income_percap'],
            'heating': ['type_heating', 'cost_heating', 'area_heating', 'time_heating'],
            'vehicle': ['vehicle_dist', 'cost_vehicle', 'vehicle_num', 'vehicle_fuel'],
            'heater': ['time_water_heater', 'freq_water_heater'],
            'cooking': ['freq_cooking', 'time_cooking', ],
            'ac': ['time_ac', 'freq_ac'],
            'demographic': ['if_single_elderly', 'if_singleAE', 'if_singleA',
                            'if_couple_elderly', 'if_coupleA', 'if_singleWithChildren',
                            'if_coupleWithChildren', 'if_grandparentKids', 'if_bigFamily',
                            'children_num', 'elderly_num'],
            'energy': ['en_total', 'en_house_heating', 'en_cooking', 'en_water_heating']}
    tag = {'cluster': 'all', 'cluster_rural': 'rural', 'cluster_urban': 'urban'}

    exp = pd.DataFrame()
    # columns = [('cluster', )]
    columns = ['cluster']
    for a, k in keys.items():
        # multi-columns
        # columns += [(a, i) for i in k]
        columns += k + [a]
        # for all samples
        g = d[k + [cluster_key]].groupby(cluster_key).mean().round(4)
        g.index = [f'{tag[cluster_key]}:{i}' for i in g.index]
        # create a blank column for separating
        g[a] = 0
        exp = pd.concat([exp, g], axis=1)

    exp = exp.reset_index()
    # exp.columns = pd.MultiIndex.from_tuples(columns)
    exp.columns = columns
    return columns[1:], exp


if __name__ == '__main__':
    import numpy as np
    from pathlib import Path

    # saving path
    path = Path('data') / 'clustered-0223'

    # specify a right version of clustered data
    clusterfile = path / 'cluster-0.27.xlsx'
    cluster = pd.read_excel(clusterfile)
    cluster = cluster.replace(-99, np.nan)

    # export grouped data with format
    _, a = create_matrix(cluster, 'cluster')
    _, r = create_matrix(cluster, 'cluster_rural')
    keys, u = create_matrix(cluster, 'cluster_urban')
    hline = pd.DataFrame([0] * a.shape[1]).T
    hline.columns = a.columns
    result = pd.concat([a, hline, r, hline, u], axis=0, ignore_index=True).reset_index(drop=True)

    # highlight results
    for i, k in enumerate(keys):
        if i == 0:
            stl = result.style.background_gradient(cmap='Purples', subset=k, axis=None)
        else:
            stl = stl.background_gradient(cmap='Purples', subset=k, axis=None)

    stl.to_excel(path / 'typology-matrix.xlsx')
