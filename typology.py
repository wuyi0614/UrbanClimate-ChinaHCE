# The script for typology identification
#

import pandas as pd


def create_matrix(d: pd.DataFrame, cluster_key: str):
    keys = {'general': ['region', 'prefecture_id', 'size', 'age', 'outside', 'live_days', 'house_area'],
            'economic': ['raw_income', 'log_raw_income', 'expenditure', 'log_expenditure', 'income_percap',
                         'log_income_percap'],
            'heating': ['type_heating', 'cost_heating', 'area_heating', 'time_heating'],
            'vehicle': ['vehicle_use', 'vehicle_dist', 'fuel_price_vehicle', 'fuel_vehicle', 'emit_vehicle',
                        'cost_vehicle', 'vehicle_num', 'vehicle_fuel'],
            'heater': ['num_water_heater', 'time_water_heater', 'freq_water_heater', 'label_water_heater'],
            'cooking': ['freq_cooking', 'time_cooking', 'num_cooking', 'power_cooking'],
            'ac': ['time_ac', 'freq_ac', 'power_ac', 'num_ac', 'label_ac'],
            'demographic': ['IF_singleAE', 'IF_singleA', 'IF_coupleA', 'IF_singleWithChildren', 'IF_coupleWithChildren',
                            'IF_single_elderly', 'IF_couple_elderly', 'IF_existElderly', 'elderNumber',
                            'childrenNumber', 'IF_grandparentKids', 'IF_bigFamily']
            }
    tag = {'cluster': 'all', 'cluster_rural': 'rural', 'cluster_urban': 'urban'}

    exp = pd.DataFrame()
    # columns = [('cluster', )]
    columns = ['cluster']
    for a, k in keys.items():
        # multi-columns
        # columns += [(a, i) for i in k]
        columns += k
        # for all samples
        g = d[k + [cluster_key]].groupby(cluster_key).mean().round(4)
        g.index = [f'{tag[cluster_key]}:{i}' for i in g.index]
        exp = pd.concat([exp, g], axis=1)

    exp = exp.reset_index()
    # exp.columns = pd.MultiIndex.from_tuples(columns)
    exp.columns = columns
    return columns[1:], exp


if __name__ == '__main__':
    import seaborn as sns

    from pathlib import Path

    # saving path
    path = Path('data') / 'clustered-0223'

    # specify a right version of clustered data
    clusterfile = Path('data') / 'clustered-0223' / 'cluster-all-0019.xlsx'
    cluster = pd.read_excel(clusterfile)

    # export grouped data with format
    _, a = create_matrix(cluster, 'cluster')
    _, r = create_matrix(cluster, 'cluster_rural')
    keys, u = create_matrix(cluster, 'cluster_urban')
    result = pd.concat([a, r, u], axis=0).reset_index(drop=True)

    # highlight results
    cmap = sns.diverging_palette(5, 250, as_cmap=True)
    for i, k in enumerate(keys):
        if i == 0:
            stl = result.style.background_gradient(cmap=cmap, subset=k, axis=None)
        else:
            stl = stl.background_gradient(cmap=cmap, subset=k, axis=None)

    stl.to_excel(path / 'typology-matrix.xlsx')
