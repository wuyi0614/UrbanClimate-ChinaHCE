# Replicable script for the first section of Results
#
# Created on 7 March 2024, by Yixuan Zhang. Modified by Yi.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from config import WSJ, VAR_MAPPING
from config import CLUSTER_MAPPING_2024 as CLUSTER_MAPPING

# assign colors to groups
COLORS = {'LED-G': 'darkred',
          'CDE': 'darkgreen',
          'DHY': 'darkblue',
          'HDN': 'lightblue',
          'LED-E': 'lightred',
          'WDS': 'lightgreen',
          'LED': 'purple'}


def scale(data: pd.DataFrame, vars: list = []):
    """data scaling"""
    scaler = MinMaxScaler()
    scaled = data.copy(True)
    for k in tqdm(vars):
        var = scaled[k].fillna(0)
        scored = scaler.fit_transform(var.values.reshape(len(scaled), 1))
        scaled[k] = scored.reshape(len(scaled), )

    return scaled


def get_data(data: pd.DataFrame, keys: list):
    zero_unsafe_keys = ['freq_cooking', 'time_cooking', 'power_cooking',
                        'time_water_heater', 'freq_water_heater', 'label_water_heater',
                        'time_ac', 'freq_ac', 'power_ac',
                        'time_heating', 'area_heating']
    for k in zero_unsafe_keys:
        data[k] = data[k].replace(0, np.nan)

    me = data[keys + ['cluster']].groupby('cluster').agg(np.nanmean)
    me.index = [CLUSTER_MAPPING['all'][i][1] for i in me.index]

    mask_urban = data['cluster_urban'].isna()
    me_urban = data.loc[~mask_urban, keys + ['cluster_urban']].groupby('cluster_urban').agg(np.nanmean)
    me_urban.index = [CLUSTER_MAPPING['urban'][i][1] for i in me_urban.index]

    mask_urban = data['cluster_rural'].isna()
    me_rural = data.loc[~mask_urban, keys + ['cluster_rural']].groupby('cluster_rural').agg(np.nanmean)
    me_rural.index = [CLUSTER_MAPPING['rural'][i][1] for i in me_rural.index]
    return me, me_urban, me_rural


def bar(data: pd.DataFrame):
    # get resident type based and cluster based data
    me, me_urban, me_rural = get_data(data, ['en_total_percap'])

    # make the figure
    fig, axes = plt.subplots(1, 3, sharey=False, figsize=(15, 8))
    fig.subplots_adjust(wspace=0.05)
    # make subplot (a), (b) and (c)
    titles = ['(a)', '(b)', '(c)']
    for idx, g in enumerate([me, me_urban, me_rural]):
        g = g.reset_index()
        g = g.sort_values('en_total_percap')
        sns.barplot(data=g, x='index', y='en_total_percap', ax=axes[idx],
                    palette=[WSJ[COLORS[i]] for i in g['index']], alpha=0.6)
        axes[idx].set_xlabel(xlabel='', fontsize=14)
        axes[idx].set_ylim(0, 1350)
        if idx == 0:
            axes[idx].set_yticks(np.arange(0, 1350, 100), fontsize=14)
            axes[idx].set_ylabel(ylabel='Per capita energy consumption (kgce)', fontsize=14)
        else:
            axes[idx].set_yticks([])
            axes[idx].set_ylabel(ylabel='', fontsize=14)

        axes[idx].set_title(f'{titles[idx]}', loc='center', fontsize=14, y=-0.05)
        axes[idx].set(xlabel=None)
        axes[idx].set_xticks([])

    patches = []
    for label, color in COLORS.items():
        patches += [mpatches.Patch(color=WSJ[COLORS[label]], label=label, alpha=0.6)]

    plt.legend(handles=patches, bbox_to_anchor=(1, 1))
    plt.savefig('data/img-0223/percap-energy-by-cluster.png', dpi=200, bbox_inches='tight')
    plt.show()
    return me, me_urban, me_rural


def heatmap(data: pd.DataFrame):
    # list variables that should be summarised
    general = ['region', 'size', 'age', 'elderly_num', 'children_num', 'if_grandparentKids',
               'outside', 'raw_income', 'expenditure']
    cooking = ['freq_cooking', 'time_cooking', 'power_cooking']
    temperature = ['num_water_heater', 'time_water_heater', 'freq_water_heater', 'label_water_heater',
                   'num_ac', 'time_ac', 'freq_ac', 'power_ac',
                   'time_heating', 'area_heating']
    vehicle = ['vehicle_num', 'vehicle_dist']
    compress = ['en_ac', 'en_computer', 'en_freezing', 'en_laundry', 'en_lighting', 'en_television']
    components = ['en_appliance', 'en_cooking', 'en_house_heating', 'en_vehicle', 'en_water_heating']
    data['en_appliance'] = data[compress].sum(axis=1)  # compress 6 types into 1 type
    # cast mean-value array
    energy = []
    for c in components:
        energy += [f'{c}_percent']
        data[energy[-1]] = data[c] / data['en_total']

    # NB. the final list of variables
    vscale = general + cooking + temperature + vehicle + energy
    me, me_urban, me_rural = get_data(data, vscale)
    me = scale(me, vscale)
    me_urban = scale(me_urban, vscale)
    me_rural = scale(me_rural, vscale)

    # restore x-axis labels
    xlabels = [VAR_MAPPING[i] for i in vscale]

    titles = ['(c)', '(b)', '(a)']
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 10))
    for idx, g in enumerate([me_rural, me_urban, me]):
        # make the figure
        im = sns.heatmap(g, cmap="RdYlGn_r", cbar=False, ax=axes[idx], linewidth=1, alpha=0.8)
        axes[idx].set_yticklabels(labels=g.index, rotation=60, fontsize=14)
        axes[idx].set_ylabel(ylabel=f'{titles[idx]}', rotation=90, fontsize=14, labelpad=12)
        axes[idx].set_xticklabels(labels=xlabels, fontsize=14, rotation=90, horizontalalignment='center')

    mappable = im.get_children()[0]
    cbar = plt.colorbar(mappable, ax=axes, pad=0.02, orientation='vertical')
    cbar.ax.tick_params(rotation=90, labelsize=14)
    plt.savefig('data/img-0223/heatmap.png', dpi=200, bbox_inches='tight')
    plt.show()
    return me, me_urban, me_rural


if __name__ == '__main__':
    # fetch the latest clustered datafile
    clusterfile = Path('data') / 'clustered-0223' / 'cluster-0.06.xlsx'
    data = pd.read_excel(clusterfile)

    # make figures
    m, u, r = bar(data)
    m, u, r = heatmap(data)

    # summarise data
    keys = ['freq_cooking', 'time_cooking', 'power_cooking',
            'num_water_heater', 'time_water_heater', 'freq_water_heater', 'label_water_heater',
            'num_ac', 'time_ac', 'freq_ac', 'power_ac', 'label_ac',
            'time_heating', 'area_heating',
            'en_vehicle', 'en_cooking', 'en_total']
    m, u, r = get_data(data, keys=keys)
