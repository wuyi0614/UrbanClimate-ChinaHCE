import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


if __name__ == '__main__':
    # load data and tests
    from pathlib import Path
    datafile = Path('output_data') / 'cluster_output-1104.csv'
    data = pd.read_csv(datafile)
    datafile = Path('output_data') / 'classification_metrics-1104.csv'
    classification_metrics = pd.read_csv(datafile)

    data['new cluster'] = 9
    index = data['cluster'] == 0
    data.loc[index, ['new cluster']] = 3
    index = data['cluster'] == 1
    data.loc[index, ['new cluster']] = 0
    index = data['cluster'] == 2
    data.loc[index, ['new cluster']] = 2
    index = data['cluster'] == 3
    data.loc[index, ['new cluster']] = 1

    n_cluster = 4

    # Combination_plot - 10 sources of energy use:
    select_var = ['en_ac', 'en_computer', 'en_cooking', 'en_freezing', 'en_heating', 'en_laundry',
                  'en_lighting', 'en_television', 'en_vehicle', 'en_waterheating']
    labels_name = ['ac', 'computer', 'cooking', 'freezing', 'heating', 'laundry',
                   'lighting', 'television', 'vehicle', 'waterheating']
    cluster = ['new cluster']
    cluster_name = ['0', '1', '2', '3']
    plot_data = data[select_var + cluster]
    percent_data = plot_data.groupby('new cluster').sum()
    for n in range(0, n_cluster):
        select_data = plot_data[plot_data['new cluster'] == n]
        plot_percent = np.array(
            select_data[select_var].apply(lambda x: x.sum()) / sum(select_data[select_var].apply(lambda x: x.sum())))
        percent_data.iloc[n, :] = plot_percent

    rank_data = percent_data.copy()
    for n in range(0, (rank_data.shape[1])):
        rank_data.iloc[:, n] = np.argsort(rank_data.iloc[:, n])

    rank_data = 4 - rank_data
    rank_label = ['1', '2', '3','4']
    fig, axes = plt.subplots()
    x = ['0', '1', '2', '3']

    fig = plt.figure()
    axes = fig.add_gridspec(2, 4)
    axes_list = []
    n = 0
    axes1 = fig.add_subplot(axes[0, n])
    axes1.pie(x=percent_data.iloc[n, :])
    axes1.set_title(('cluster %s' % n))

    n = 1
    axes2 = fig.add_subplot(axes[0, n])
    axes2.pie(x=percent_data.iloc[n, :])
    axes2.set_title(('cluster %s' % n))

    n = 2
    axes3 = fig.add_subplot(axes[0, n])
    axes3.pie(x=percent_data.iloc[n, :])
    axes3.set_title(('cluster %s' % n))

    n = 3
    axes4 = fig.add_subplot(axes[0, n])
    axes4.pie(x=percent_data.iloc[n, :])
    axes4.set_title(('cluster %s' % n))

    # Put a legend to the right of the current axis
    axes4.legend(labels=labels_name, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0., fontsize=8)

    axes5 = fig.add_subplot(axes[1, 0:4])
    axes5.imshow(rank_data)
    axes5.set_xticks(range(0, len(labels_name), 1))
    axes5.set_xticklabels(labels=labels_name, rotation=45, fontsize=8)
    axes5.set_yticks(range(0, len(cluster_name), 1))
    axes5.set_yticklabels(cluster_name)
    axes5.set_xlabel("Energy type")
    axes5.set_ylabel("Cluster label")
    axes5.set_title('Heat plot')

    for i in range(len(labels_name)):
        for j in range(len(cluster_name)):
            axes5.text(i, j, int(rank_data.iat[j, i]),
                            horizontalalignment="center", verticalalignment="center", color="w")

    # Put a legend to the right of the current axis
    plt.suptitle('Energy Use Distribution')
    plt.show()

    # Combination_plot - 8 regions:
    select_var = ['region_codes']
    labels_name = ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
    cluster = ['new cluster']
    cluster_name = ['0', '1', '2', '3']
    plot_data = data[select_var + cluster]
    percent_data = pd.DataFrame(columns=labels_name, index=cluster_name, dtype=float)
    for n in range(0, n_cluster):
        select_data = plot_data[plot_data['new cluster'] == n]
        plot_percent = select_data.groupby(select_var).count()/select_data.groupby(select_var).count().sum()
        percent_data.iloc[n, :] = plot_percent['new cluster']

    rank_data = percent_data.copy()
    for n in range(0, (rank_data.shape[1])):
        rank_data.iloc[:, n] = np.argsort(rank_data.iloc[:, n])

    rank_data = 4 - rank_data
    rank_label = ['1', '2', '3', '4']
    fig, axes = plt.subplots()

    fig = plt.figure()
    axes = fig.add_gridspec(2, 4)
    axes_list = []
    n = 0
    axes1 = fig.add_subplot(axes[0, n])
    p1 = axes1.pie(x=percent_data.iloc[n, :])
    axes1.set_title(('cluster %s' % n))

    n = 1
    axes2 = fig.add_subplot(axes[0, n])
    p2 = axes2.pie(x=percent_data.iloc[n, :])
    axes2.set_title(('cluster %s' % n))

    n = 2
    axes3 = fig.add_subplot(axes[0, n])
    p3 = axes3.pie(x=percent_data.iloc[n, :])
    axes3.set_title(('cluster %s' % n))

    n = 3
    axes4 = fig.add_subplot(axes[0, n])
    p4 = axes4.pie(x=percent_data.iloc[n, :])
    axes4.set_title(('cluster %s' % n))

    # Put a legend to the right of the current axis
    axes4.legend(labels=labels_name, loc='center left', bbox_to_anchor=(1.05,0.5),borderaxespad = 0., fontsize=8)


    axes5 = fig.add_subplot(axes[1, 0:4])
    axes5.imshow(rank_data)
    axes5.set_xticks(range(0, len(labels_name), 1))
    axes5.set_xticklabels(labels=labels_name, rotation=45, fontsize=8)
    axes5.set_yticks(range(0, len(cluster_name), 1))
    axes5.set_yticklabels(cluster_name)
    axes5.set_xlabel("Region")
    axes5.set_ylabel("Cluster label")
    axes5.set_title('Heat plot')

    for i in range(len(labels_name)):
        for j in range(len(cluster_name)):
            axes5.text(i, j, int(rank_data.iat[j, i]),
                       horizontalalignment="center", verticalalignment="center", color="w")

    # Put a legend to the right of the current axis
    plt.suptitle('Region Distribution')
    plt.show()

    # Combination_plot - 12 demographic:
    select_var = ['demographicType']
    labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
    cluster = ['new cluster']
    cluster_name = ['0', '1', '2', '3']
    plot_data = data[select_var + cluster]
    percent_data = pd.DataFrame(columns=labels_name, index=cluster_name, dtype=float)
    for n in range(0, n_cluster):
        select_data = plot_data[plot_data['new cluster'] == n]
        plot_percent = select_data.groupby(select_var).count() / select_data.groupby(select_var).count().sum()
        percent_data.iloc[n, :] = plot_percent['new cluster']

    rank_data = percent_data.copy()
    for n in range(0, (rank_data.shape[1])):
        rank_data.iloc[:, n] = np.argsort(rank_data.iloc[:, n])

    rank_data = 4 - rank_data
    rank_label = ['1', '2', '3', '4']
    fig, axes = plt.subplots()

    fig = plt.figure()
    axes = fig.add_gridspec(2, 4)
    axes_list = []
    n = 0
    axes1 = fig.add_subplot(axes[0, n])
    p1 = axes1.pie(x=percent_data.iloc[n, :])
    axes1.set_title(('cluster %s' % n))

    n = 1
    axes2 = fig.add_subplot(axes[0, n])
    p2 = axes2.pie(x=percent_data.iloc[n, :])
    axes2.set_title(('cluster %s' % n))

    n = 2
    axes3 = fig.add_subplot(axes[0, n])
    p3 = axes3.pie(x=percent_data.iloc[n, :])
    axes3.set_title(('cluster %s' % n))

    n = 3
    axes4 = fig.add_subplot(axes[0, n])
    p4 = axes4.pie(x=percent_data.iloc[n, :])
    axes4.set_title(('cluster %s' % n))

    # Put a legend to the right of the current axis
    axes4.legend(labels=labels_name, loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0., fontsize=8)

    axes5 = fig.add_subplot(axes[1, 0:4])
    axes5.imshow(rank_data)
    axes5.set_xticks(range(0, len(labels_name), 1))
    axes5.set_xticklabels(labels=labels_name, rotation=45, fontsize=8)
    axes5.set_yticks(range(0, len(cluster_name), 1))
    axes5.set_yticklabels(cluster_name)
    axes5.set_xlabel("Demographic Type")
    axes5.set_ylabel("Cluster label")
    axes5.set_title('Heat plot')

    for i in range(len(labels_name)):
        for j in range(len(cluster_name)):
            axes5.text(i, j, int(rank_data.iat[j, i]),
                       horizontalalignment="center", verticalalignment="center", color="w")

    # Put a legend to the right of the current axis
    plt.suptitle('Demographic Type Distribution')
    plt.show()


    # heat plot
    fig = plt.figure()
    axes = fig.add_gridspec()
    axes5 = fig.add_subplot()
    axes5.imshow(classification_metrics)
    labels_name = classification_metrics.columns
    cluster_name = ['0', '1', '2', '3']
    axes5.set_xticks(range(0, len(labels_name), 1))
    axes5.set_xticklabels(labels=labels_name, rotation=45, fontsize=8)
    axes5.set_yticks(range(0, len(cluster_name), 1))
    axes5.set_yticklabels(cluster_name)
    axes5.set_xlabel("Demographic Type")
    axes5.set_ylabel("Cluster label")
    axes5.set_title('Heat plot')

    for i in range(len(labels_name)):
        for j in range(len(cluster_name)):
            axes5.text(i, j, format(classification_metrics.iat[j, i], '.2f'),
                       horizontalalignment="center", verticalalignment="center", color="w")

    # Put a legend to the right of the current axis
    plt.suptitle('Demographic Type Distribution')
    plt.show()



    # Pie plot - fuel:
    select_var = ['fuel1','fuel2','fuel3','fuel4','fuel5',
                  'fuel6','fuel7','fuel8','fuel9','fuel10']
    cluster = ['cluster']
    plot_data = data[select_var+cluster]

    fig, axes = plt.subplots(1,n_cluster,figsize=(16, 4))
    axes_list = []

    for n in range(0,n_cluster):
        select_data = plot_data[plot_data['cluster'] == n]
        plot_percent = np.array(
            select_data[select_var].apply(lambda x: x.sum()) / sum(select_data[select_var].apply(lambda x: x.sum())))
        # colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(plot_percent)))
        axes[n].pie(x=plot_percent)
        axes[n].set_title(('cluster %s' % n))

    fig.legend(labels=select_var, loc='right')
    plt.suptitle('Fuel Type Distribution')
    plt.show()

    # Pie plot - demographicType:
    select_var = ['demographicType']
    cluster = ['cluster']
    plot_data = data[select_var + cluster]

    fig, axes = plt.subplots(1, n_cluster, figsize=(16, 4))
    axes_list = []

    for n in range(0, n_cluster):
        select_data = plot_data[plot_data['cluster'] == n]
        plot_percent = select_data.groupby(select_var).count()
        labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(plot_percent)))
        axes[n].pie(x=plot_percent['cluster'])
        axes[n].set_title(('cluster %s' % n))

    fig.legend(labels=labels_name, loc='right')
    plt.suptitle('DemographicType Distribution')
    plt.show()

    # Pie plot - IF_single_elderly:
    select_var = ['IF_single_elderly']
    cluster = ['cluster']
    plot_data = data[select_var + cluster]

    select_data = plot_data[plot_data['IF_single_elderly'] == 0]
    plot_percent_0 = select_data.groupby('cluster').count()
    select_data = plot_data[plot_data['IF_single_elderly'] == 1]
    plot_percent_1 = select_data.groupby('cluster').count()
    x = ['0','1','2','3']
    y1 = plot_percent_0 / (plot_percent_0 + plot_percent_1)
    y2 = plot_percent_1 / (plot_percent_0 + plot_percent_1)
    y1 = list(y1['IF_single_elderly'])
    y2 = list(y2['IF_single_elderly'])
    plt.bar(x, y1, width=0.4, label='0', color='#f9766e', edgecolor='grey', zorder=5)
    plt.bar(x, y2, width=0.4, bottom=y1, label='1', color='#00bfc4', edgecolor='grey', zorder=5)
    plt.tick_params(axis='x', length=0)
    plt.xlabel('Site', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.ylim(0, 1.01)
    plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
    plt.grid(axis='y', alpha=0.5, ls='--')
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    # plt.savefig('bar2.png', dpi=600)
    plt.show()


    # Pie plot - IF_bigFamily:
    select_var = ['IF_bigFamily']
    cluster = ['cluster']
    plot_data = data[select_var + cluster]

    select_data = plot_data[plot_data['IF_bigFamily'] == 0]
    plot_percent_0 = select_data.groupby('cluster').count()
    select_data = plot_data[plot_data['IF_bigFamily'] == 1]
    plot_percent_1 = select_data.groupby('cluster').count()
    x = ['0', '1', '2', '3']
    y1 = plot_percent_0 / (plot_percent_0 + plot_percent_1)
    y2 = plot_percent_1 / (plot_percent_0 + plot_percent_1)
    y1 = list(y1['IF_bigFamily'])
    y2 = list(y2['IF_bigFamily'])
    plt.bar(x, y1, width=0.4, label='0', color='#f9766e', edgecolor='grey', zorder=5)
    plt.bar(x, y2, width=0.4, bottom=y1, label='1', color='#00bfc4', edgecolor='grey', zorder=5)
    plt.tick_params(axis='x', length=0)
    plt.xlabel('Site', fontsize=12)
    plt.ylabel('Concentration', fontsize=12)
    plt.ylim(0, 1.01)
    plt.yticks(np.arange(0, 1.2, 0.2), [f'{i}%' for i in range(0, 120, 20)])
    plt.grid(axis='y', alpha=0.5, ls='--')
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    # plt.savefig('bar2.png', dpi=600)
    plt.show()

    # # Pie plot - 10 sources of energy use:
    # select_var = ['en_ac','en_computer', 'en_cooking', 'en_freezing', 'en_heating', 'en_laundry',
    #           'en_lighting', 'en_television', 'en_vehicle', 'en_waterheating']
    # cluster = ['new cluster']
    #
    # plot_data = data[select_var+cluster]
    #
    # fig, axes = plt.subplots(1,n_cluster,figsize=(15, 3))
    # axes_list = [labels_name]
    #
    # percent_data = plot_data.groupby('new cluster').sum()
    # for n in range(0,n_cluster):
    #     select_data = plot_data[plot_data['new cluster'] == n]
    #     plot_percent = np.array(select_data[select_var].apply(lambda x:x.sum()) / sum(select_data[select_var].apply(lambda x:x.sum())))
    #     percent_data.iloc[n,:] = plot_percent
    #     # colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(plot_percent)))
    #     axes[n].pie(x=plot_percent)
    #     axes[n].set_title(('cluster %s' % n))
    #
    # fig.legend(labels=select_var, loc='right')
    # plt.suptitle('Energy Use Distribution')
    # plt.show()
    #
    # ## bump plot
    # rank_data = percent_data
    # for n in range(0, (rank_data.shape[1])):
    #     rank_data.iloc[:,n] = np.argsort(rank_data.iloc[:,n])
    #
    # fig, axes = plt.subplots()
    # x = ['0','1','2','3']
    # axes.plot(x,rank_data.iloc[:,n])
    # plt.show()
    #
    # # Pie plot - region:
    # select_var = ['region_codes']
    # cluster = ['cluster']
    # plot_data = data[select_var+cluster]
    #
    # fig, axes = plt.subplots(1,n_cluster,figsize=(15, 4))
    # axes_list = []
    #
    # for n in range(0,n_cluster):
    #     select_data = plot_data[plot_data['cluster'] == n]
    #     plot_percent = select_data.groupby(select_var).count()
    #     labels_name = ['0','1','2','3','4','5','6','7']
    #     # colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(plot_percent)))
    #     axes[n].pie(x=plot_percent['cluster'])
    #     axes[n].set_title(('cluster %s' % n))
    #
    # fig.legend(labels=labels_name, loc='right')
    # plt.suptitle('Region Distribution')
    # plt.show()