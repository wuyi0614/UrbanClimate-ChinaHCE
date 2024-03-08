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
from config import CLUSTER_MAPPING_2024 as CLUSTER_MAPPING
from config import WSJ
from result_one import COLORS


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


def inequality_no_cluster(data: pd.DataFrame, save: Path):
    """
    Use specific dependent variable for inequality analysis

    :param data: the original dataset
    :param save: saving path
    """
    # we make the figure with 2 subplots
    # subplot (a)
    arr1 = [data.loc[data.resident == 1, 'en_total'].values,
            data.loc[data.resident == 0, 'en_total'].values,
            data.loc[data.region == 1, 'en_total'].values,
            data.loc[data.region == 0, 'en_total'].values]
    label1 = ['Urban', 'Rural', 'South', 'North']
    # subplot (b)
    compress = ['en_ac', 'en_computer', 'en_freezing', 'en_laundry', 'en_lighting', 'en_television']
    components = ['en_appliance', 'en_cooking', 'en_house_heating', 'en_vehicle', 'en_water_heating']
    label2 = ['Appliance', 'Cooking', 'Space heating', 'Vehicle', 'Water heating']
    data['en_appliance'] = data[compress].sum(axis=1)  # compress 6 types into 1 type
    arr2 = [data[k].dropna().values for k in components]
    # make the figure
    titles = ['(a)', '(b)']
    gi = []
    fig, axes = plt.subplots(1, 2, sharey=False, figsize=(16, 8))
    for idx, (label, array) in enumerate(zip([label1, label2], [arr1, arr2])):
        for i, a in enumerate(array):
            g = gini(a)
            gi += [g]
            lo = lorenz(a)
            # we need the X values to be between 0.0 to 1.0
            c = list(WSJ.values())[i]
            axes[idx].plot([0, 1], [0, 1], linewidth=1.5, color='#999999')
            axes[idx].plot(np.linspace(0.0, 1.0, lo.size), lo, linewidth=1.5, linestyle=':', color=c,
                           label=f'{label[i]} ({"%.3f" % g})')
            ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
            axes[idx].set_xticklabels(ticks, size=12)
            axes[idx].set_yticklabels(ticks, size=12)
            axes[idx].set_ylabel('Cumulative share of energy consumption', size=14)
            axes[idx].set_xlabel('Cumulative share of households', size=14)
            axes[idx].set_title(f'{titles[idx]}', loc='center', fontsize=14, y=-.135)
            axes[idx].legend(fontsize=14)

    plt.savefig(save / 'inequality-region+energies.png', dpi=200, bbox_inches='tight')
    plt.show()
    return gi


def inequality_with_cluster(data: pd.DataFrame, save: Path):
    """Inequality analysis after clustering. See documentation in inequality_no_cluster()."""
    # all GINI
    arr1 = [g['en_total'].values for _, g in data.groupby('cluster')]
    label1 = [CLUSTER_MAPPING['all'][l][1] for l, _ in cluster.groupby('cluster')]

    # urban GINI
    urban = data[~data.cluster_urban.isna()].groupby('cluster_urban')
    arr2, label2 = [], []
    for l, g in urban:
        arr2 += [g['en_total'].values]
        c = CLUSTER_MAPPING['urban'][l][1]
        label2 += [f'{c}']

    # rural GINI
    rural = data[~data.cluster_rural.isna()].groupby('cluster_rural')
    arr3, label3 = [], []
    for l, g in rural:
        arr3 += [g['en_total'].values]
        c = CLUSTER_MAPPING['rural'][l][1]
        label3 += [f'{c}']

    # make the figure with 3 subplots
    titles = ['(a)', '(b)', '(c)']
    gi = []
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(21, 7))
    for idx, (label, array) in enumerate(zip([label1, label2, label3], [arr1, arr2, arr3])):
        for i, a in enumerate(array):
            g = gini(a)
            gi += [g]
            lo = lorenz(a)
            # we need the X values to be between 0.0 to 1.0
            c = WSJ[COLORS[label[i]]]
            axes[idx].plot([0, 1], [0, 1], linewidth=1.5, color='#999999')
            axes[idx].plot(np.linspace(0.0, 1.0, lo.size), lo, linewidth=1.5, linestyle=':', color=c,
                           label=f'{label[i]} ({"%.3f" % g})')
            ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)
            axes[idx].set_xticklabels(ticks, size=12)
            axes[idx].set_yticklabels(ticks, size=12)
            axes[idx].set_ylabel('Cumulative share of energy consumption', size=14)
            axes[idx].set_xlabel('Cumulative share of households', size=14)
            axes[idx].set_title(f'{titles[idx]}', loc='center', fontsize=14, y=-.135)
            axes[idx].legend(fontsize=14)

    plt.savefig(save / 'inequality-clustered.png', dpi=200, bbox_inches='tight')
    plt.show()
    return gi


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
    save = Path('data') / 'img-0223'
    save.mkdir(exist_ok=True)

    # load two datasets
    datafile = Path('data') / 'mergedata-0229.xlsx'
    nocluster = pd.read_excel(datafile, engine='openpyxl')

    datafile = Path('data') / 'clustered-0223' / 'cluster-0.06.xlsx'
    cluster = pd.read_excel(datafile)

    # preprocessing and criteria filtering
    data = nocluster[nocluster['en_total'] > 0]
    print(f'Got {len(data)} and removed {len(nocluster) - len(data)} records!')

    # the general GINI coef of all samples before clustering
    g = gini(data['en_total'].values)
    print(f'General GINI: {g}')

    # compute GINI coefs for urban, rural, north, south and components
    gi_no_cluster = inequality_no_cluster(data, save)

    # compute GINI coefs for all, urban, rural samples with clusters
    gi_cluster = inequality_with_cluster(cluster, save)
    for i in [(0, 6), (6, 11), (11, 16)]:
        g = gi_cluster[i[0]: i[1]]
        print(np.mean(g))