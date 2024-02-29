# Replicable script for the second section of Results
#
# Created on 31 Oct 2023, by Yi
#

# In this section, we will basically use GINI index and xxx to measure the inequality between
# regions, cities, and groups

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from config import WSJ, CLUSTER_MAPPING


def gini(arr):
    cu = np.cumsum(sorted(np.append(arr, 0)))
    su = cu[-1]
    x = np.array(range(0, len(cu))) / float(len(cu) - 1)
    y = cu / su
    B = np.trapz(y, x=x)
    A = 0.5 - B
    return A / (A + B)


def lorenz(arr):
    # this divides the prefix sum by the total sum
    # this ensures all the values are between 0 and 1.0
    arr.sort()
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return np.insert(scaled_prefix_sum, 0, 0)


def inequality(*arrays, labels: list, name: str, save: Path):
    """
    Plot the inequality result using Lorenz curves

    :param arrays: a set of arrays for calculation
    :param labels: labels shown in the legend
    :param name: save figure by the name
    :param save: saving path, change it every time re-run the script
    """
    fig = plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linewidth=1.5, color='#999999')
    plt.ylabel('Cumulative share of energy consumption', size=12)
    plt.xlabel('Cumulative share of households', size=12)

    for i, a in enumerate(arrays):
        g = gini(a)
        lo = lorenz(a)
        # we need the X values to be between 0.0 to 1.0
        c = list(WSJ.values())[i]
        plt.plot(np.linspace(0.0, 1.0, lo.size), lo, linewidth=1.5, linestyle=':', color=c,
                 label=f'{labels[i]} ({round(g, 3)})')
        # plot the straight line perfect equality curve

    # output the figure
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    plt.margins(0.0)
    plt.legend(loc=0, fontsize=10)
    fig.savefig(save / f'inequality-{name}.pdf', format='pdf', dpi=200)
    plt.show()


if __name__ == '__main__':
    # configure saving path
    path = Path('data') / 'img-0223'
    path.mkdir(exist_ok=True)

    # load data
    datafile = Path('data') / 'mergedata-1114.xlsx'
    raw = pd.read_excel(datafile, engine='openpyxl')
    # preprocessing and criteria filtering
    data = raw[raw['en_total'] > 0]
    print(f'Got {len(data)} and removed {len(raw) - len(data)} records!')

    # compute GINI coefs for urban, rural, north, south
    inequality(data.loc[data.resident == 1, 'en_total'].values,
               data.loc[data.resident == 0, 'en_total'].values,
               data.loc[data.region == 1, 'en_total'].values,
               data.loc[data.region == 0, 'en_total'].values,
               labels=['Urban', 'Rural', 'South', 'North'], name='region', save=path)

    # compute GINI coefs for components
    compress = ['en_ac', 'en_computer', 'en_freezing', 'en_laundry', 'en_lighting', 'en_television']
    components = ['en_appliance', 'en_cooking', 'en_heating', 'en_vehicle', 'en_waterheating']
    labels = ['Appliance', 'Cooking', 'Space heating', 'Vehicle', 'Water heating']
    data['en_appliance'] = data[compress].sum(axis=1)  # compress 6 types into 1 type
    array = [data[k].values for k in components]
    inequality(*array, labels=labels, name='component', save=path)

    # GINI coefs after clustering
    clusterfile = Path('data') / 'clustered-0223' / 'cluster-all-0019.xlsx'
    cluster = pd.read_excel(clusterfile)
    cluster['en_appliance'] = cluster[compress].sum(axis=1)  # compress 6 types into 1 type
    # all GINI
    array = [g['en_total'].values for _, g in cluster.groupby('cluster')]
    labels = [CLUSTER_MAPPING['all'][l][1] for l, _ in cluster.groupby('cluster')]
    inequality(*array, labels=labels, name='cluster', save=path)
    # urban GINI
    group = cluster[~cluster.cluster_urban.isna()].groupby('cluster_urban')
    array, labels = [], []
    for l, g in group:
        array += [g['en_total'].values]
        c = CLUSTER_MAPPING['urban'][l][1]
        print(c)
        labels += [f'{c}']

    inequality(*array, labels=labels, name='urban', save=path)
    # rural GINI
    group = cluster[~cluster.cluster_rural.isna()].groupby('cluster_rural')
    array, labels = [], []
    for l, g in group:
        array += [g['en_total'].values]
        c = CLUSTER_MAPPING['rural'][l][1]
        labels += [f'{c}']

    inequality(*array, labels=labels, name='rural', save=path)

    for a, l in zip(array, labels):
        print(f'{l}: {len(a)}')
        plt.plot(a, label=l)

    plt.legend()
    plt.show()
