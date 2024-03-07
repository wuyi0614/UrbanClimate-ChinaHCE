# Clustering script
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer
from config import WSJ, VAR_MAPPING

# global config
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def preprocessing(data: pd.DataFrame, vars: list = [], scale: str = 'zscore', **param):
    """data preprocessing and standardization

    :param data: the dataframe before preprocessing
    :param vars: the list of variables that should be preprocessed
    :param scale: the optional standardization method, 'scale' or 'robust' or 'l1' or 'l2' or 'max'
    """
    if scale == 'zscore':
        func = StandardScaler(**param)
    elif scale == 'robust':
        func = RobustScaler(**param)
    else:
        func = Normalizer(norm=scale)

    scaled = data.copy(True)
    for k in tqdm(vars):
        var = scaled[k].fillna(0)
        scored = func.fit_transform(var.values.reshape(len(scaled), 1))
        scaled[k] = scored.reshape(len(scaled), )

    return scaled


def lasso_modelling(data: pd.DataFrame,
                    indep_var: dict,
                    dep_var: str,
                    alpha_range: list = None,
                    max_iteration=1000,
                    display=True,
                    **params):
    if alpha_range is None:
        alpha_range = np.linspace(0.001, 0.1, 1000)

    vars = list(indep_var.keys())
    x = data[vars].fillna(0).copy(True).values
    y = data[dep_var].copy(True).values

    model = Lasso(params.get('tol', 1e-4))  # TODO: default params could be configured for convergence
    result = GridSearchCV(model,
                          param_grid={'alpha': alpha_range, 'max_iter': [max_iteration]},
                          cv=10,
                          scoring=params.get('scoring', 'neg_mean_absolute_error'),  # might be changed
                          n_jobs=4)
    result.fit(x, y)
    print('Scoring: %.5f' % result.best_score_)
    print('Optimal param：\n', result.best_params_)
    alpha = result.best_params_

    # with optimal model
    la = Lasso(**alpha).fit(x, y)  # ... find the best alpha
    la_coef = pd.DataFrame(la.coef_, columns=["coef"])
    la_coef["vars"] = vars
    la_coef = la_coef.sort_values("coef", ascending=False)
    la_coef["colors"] = WSJ['lightred']
    la_coef.loc[la_coef.coef < 0, "colors"] = WSJ['lightgreen']

    coef = pd.concat([la_coef[la_coef.coef == 0], la_coef[la_coef.coef < 0].sort_values('coef')], axis=0)
    coef = pd.concat([coef, la_coef[la_coef.coef > 0].sort_values('coef')], axis=0)

    # output distribution of weights of variables
    if display:
        fig = plt.figure(figsize=(6, 6))
        x_range = range(len(coef))
        plt.barh(x_range, coef.coef.values, color=coef.colors.values, alpha=0.65)
        plt.vlines(0, -0.5, len(coef) - 0.5, color='grey', linewidth=0.5)

        ticks = [indep_var[v] for v in coef.vars.values]
        plt.yticks(x_range, labels=ticks, size=10, rotation=0, verticalalignment='center', horizontalalignment='right')
        plt.ylabel("Feature importance", fontsize=12)
        plt.xlabel("Household characteristics", fontsize=12)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.margins(0.01)
        plt.tight_layout()
        # plt.grid(axis="x", color="grey", alpha=0.3)
        fig.savefig('img/lasso1.pdf', format='pdf', dpi=200)
        plt.show()

    return la_coef


def clustering_modelling(data: pd.DataFrame,
                         vars: list,
                         index: str = 'id',
                         epoch: int = 10,
                         n_clusters: int = 1,
                         metric: str = 'cosine',
                         display: bool = True):
    """Modelling process for clustering"""
    # reconcile with features
    x = data[vars].fillna(0).values
    train = x.copy()  # could use fitted(post-lasso) or X (raw)

    # clustering pipeline
    param = {
        "n_clusters": n_clusters,
        "init": "k-means++",
        "algorithm": "elkan",
        "random_state": 0
    }

    if n_clusters == 1:
        eva = []
        for n in tqdm(range(2, epoch, 1)):
            # baseline: K-Means, use n_cluster = 3 as default
            param["n_clusters"] = n
            km = KMeans(**param).fit(train)
            y_pred = km.predict(train)
            eva += [[silhouette_score(train, y_pred, metric=metric), davies_bouldin_score(train, y_pred)]]

        exp = pd.DataFrame(eva, columns=["silhouette_score", "calinski_harabasz_score"])
        # for K-means, select the biggest sihouette_score
        n_clusters = exp.silhouette_score.values.argmax() + 2
        print(f"The finalised number of clusters is: {n_clusters}")

        # plot the iteration process
        if display:
            x_range = range(len(exp))
            fig = plt.figure(figsize=(10, 5), dpi=80)
            plt.plot(x_range, exp.silhouette_score.values, marker="^", color="darkgreen")
            plt.xticks(x_range, range(2, epoch, 1), size=12)

            plt.axvspan(n_clusters - 2.5, n_clusters - 1.5, 0, 0.975,
                        facecolor="none", edgecolor="red", linewidth=2, linestyle="--")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette score")  # the higher, the better
            plt.grid(axis="y", color="grey", alpha=0.3)
            plt.savefig('img/cluster1.png', dpi=200)
            plt.show()

    # rerun the clustering model
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(train)
    y_pred = km.predict(train)

    # make sure you're passing `cluster` back to the raw dataset
    data['cluster'] = y_pred

    # statistics
    counted = data[[index, 'cluster']].groupby('cluster').count()
    # print(f"Counted households: {counted}")

    # check the averaged percap emissions
    counted = data[["en_total", 'cluster']].groupby('cluster').mean()
    # print(f"Classified energy consumption: {counted}")

    # output
    return exp, data.copy(True)


def cluster_validator(data: pd.DataFrame, validate_keys: list = [], save=False):
    """Validate clustered data with certain criteria"""
    compare = pd.DataFrame()
    for i, (_, sub) in enumerate(data.groupby('cluster')):
        count = sub[validate_keys].astype(float).apply(lambda x: x.mean(), axis=0).to_frame().T
        count.loc[count.index, 'count'] = len(sub)

        # reorder the keys by, 1) validate_keys; 2) the other keys
        count['cluster'] = i
        compare = pd.concat([compare, count], axis=0)

    if save:
        compare.T.to_excel(f'data/cluster-validation-1104.xlsx')

    return compare


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # load data and tests
    from pathlib import Path

    path = Path('data') / 'clustered-0223'
    path.mkdir(exist_ok=True)

    datafile = Path('data') / 'mergedata-0229.xlsx'
    raw = pd.read_excel(datafile, engine='openpyxl')
    raw = raw.drop(
        columns=['media', 'aware_trust', 'aware_harm', 'aware_justice', 'aware_happy', 'job', 'fuel_vehicle'])

    # preprocessing and criteria filtering
    data = raw[(raw['en_total'] > 0 & ~raw['region'].isna())]
    print(f'Got {len(data)} and removed {len(raw) - len(data)} records!')

    # NB. careful about nan values as zeroes may have certain meanings and should not be used as null values
    na_safe_keys = ['children_num', 'elderly_num', 'num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
                    'num_water_heater', 'freq_water_heater', 'time_water_heater', 'num_ac', 'freq_ac', 'power_ac',
                    'time_ac', 'label_water_heater', 'label_ac', 'type_heating', 'time_heating', 'area_heating',
                    'cost_heating', 'own_vehicle', 'fuel_price_vehicle', 'cost_vehicle', 'vehicle_dist',
                    'vehicle_use', 'vehicle_fuel', 'raw_income', 'expenditure', 'income_percap']
    # TODO: missing values will affect results (especially for those rural households)
    data[na_safe_keys] = data[na_safe_keys].replace(-99, np.nan).fillna(0)

    # convert province/city id
    data['log_en_total'] = np.log(data['en_total'].values + 1)
    data['en_total_percap'] = data['en_total'].values / data['size'].values
    data['log_en_total_percap'] = np.log(data['en_total'].values / data['size'].values + 1)

    data['log_expenditure'] = np.log(data['expenditure'].values + 1)
    data['log_raw_income'] = np.log(data['raw_income'].values + 1)
    data['log_income_percap'] = np.log(data['income_percap'].values + 1)

    data['province_id'] = data['province'].apply(lambda x: list(data['province'].unique()).index(x))
    data['prefecture_id'] = data['prefecture'].apply(lambda x: list(data['prefecture'].unique()).index(x))

    # list of variables going into the LASSO/clustering models
    # NB. It's necessary to do standardization before LASSO because LASSO puts constraints on coefs by its magnitude.
    #     As for k-mean clustering, standardization is necessary because
    #     `K-means clustering is "isotropic" in all directions`

    var_geo = ['prefecture_id']  # NB. region is proxyed by heating features!
    var_demo = ['size', 'age', 'house_area', 'children_num', 'elderly_num']
    var_econ = ['expenditure', 'income_percap', 'raw_income']
    var_live = ['outside', 'live_days']
    var_family = ['if_single_elderly', 'if_singleAE', 'if_singleA',
                  'if_couple_elderly', 'if_coupleA', 'if_singleWithChildren',
                  'if_coupleWithChildren', 'if_grandparentKids', 'if_bigFamily',
                  'if_existElderly', 'if_existChildren']
    # NB. duplicated driving variables could dramatically change the results
    #       vehicle_fuel = vehicle fuel type (93/97 gasoline); vehicle_use = fuel_vehicle
    var_app = ['num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
               'num_water_heater', 'freq_water_heater', 'time_water_heater',
               'num_ac', 'freq_ac', 'power_ac', 'time_ac', 'label_water_heater', 'label_ac',
               'type_heating', 'time_heating', 'area_heating', 'cost_heating']
    # NB. changed on 2024-02-23. var_mob is separated from var_app because only a few people owning cars,
    #       std may change the pattern.
    var_mob = ['vehicle_num', 'fuel_price_vehicle', 'cost_vehicle',
               'vehicle_dist', 'vehicle_use', 'vehicle_fuel']  # NB. own_vehicle and emit_vehicle is moved
    # var_mob = ['vehicle_dist', 'vehicle_fuel']  # NB. updated on 2024-02-25.
    var_energy = ['en_ac', 'en_computer', 'en_cooking', 'en_freezing', 'en_heating',
                  'en_laundry', 'en_lighting', 'en_television', 'en_vehicle', 'en_water_heating']

    # Changelog: before 2024-02-23, var_std = var_demo + var_econ + var_app + var_live + var_mob
    # The reason why var_family should not be standardized because it's a sparse matrix
    # var_std = var_geo + var_demo + var_econ + var_app + var_live

    # Changelog: before 2024-02-23, vars_all = var_geo + var_demo + var_econ + var_app + var_mob + var_live + var_family
    vars_all = var_geo + var_demo + var_econ + var_app + var_live + var_mob + var_family

    # training dataset after preprocessing
    # Changelog: Changed on 2024-02-26, it used to be vars_std (not all vars)
    train = preprocessing(data.copy(True), vars=vars_all + ['en_total_percap'], scale='zscore')
    vv = []
    for v in [var_geo, var_demo, var_econ, var_app, var_mob, var_live, var_family]:
        vv += v
        vars_map = {k: VAR_MAPPING[k] for k in vv}
        var_lasso = lasso_modelling(train, indep_var=vars_map, dep_var='log_en_total_percap',
                                    tol=1e-4, max_iteration=100, scoring='neg_mean_absolute_error')

        v = var_lasso.loc[var_lasso['coef'].abs() > 0.07, 'vars'].values.tolist()
        score, cls = clustering_modelling(data, vars=v, epoch=13, n_clusters=1, display=False)
        print(score)

    # Changelog: export describe() for checking purpose
    pd.concat([
        data[vars_all].describe(),
        pd.DataFrame(np.zeros(len(vars_all)).reshape(1, len(vars_all)), columns=vars_all),
        train[vars_all].describe()
    ], axis=0).to_excel(path / 'summary-checking.xlsx')

    # feature engineering with LASSO
    vars_map = {k: VAR_MAPPING[k] for k in vars_all}
    # NB. make sure the LASSO converge
    var_lasso = lasso_modelling(train, indep_var=vars_map, dep_var='en_total_percap',
                                tol=1e-4, max_iteration=100, scoring='neg_mean_absolute_error')
    print(var_lasso)

    # run clustering: record K values and Silhouette scores for a heatmap selection
    silhouette = pd.DataFrame()
    # use the min/max of LASSO coefficients to create grids for searching
    for i in np.linspace(0.0, var_lasso['coef'].abs().max(), num=20, endpoint=False):
        vars = var_lasso.loc[var_lasso['coef'].abs() >= i, 'vars'].values.tolist()
        tag = str(round(i, 2))  # tag for output
        if len(vars) <= 3:
            print('Clustering is over due to an empty variable list!')
            break

        print(f'Clustering with {vars}!')

        # Notably, the reason we do not use `Gap Statistics` for the optimal K selection is that it only works
        # in 2-D cluster, and in our case, the biggest issue is the deviation between variables is too large, which
        # results in very bad clustered results. If the data is preprocessed by T-SNE, and it is clustered with the same
        # approach, we can at least obtain K=6.

        # visualisation of clustered data points
        # tsne = TSNE(n_components=2)
        # result = tsne.fit_transform(train[var_energy])
        # plot_embedding(result, cls['cluster'], 't-SNE')  # cls['cluster'].values

        # KMeans with standardised and log-transformed data
        # NB. n_cluseters=1 means it needs experiment
        score, cls = clustering_modelling(train, vars=vars, epoch=13, n_clusters=1, display=False)

        # clustering validation
        valid_keys = ['en_cooking_percap', 'en_heating_percap',
                      'en_vehicle_percap', 'en_waterheating_percap']
        data['cluster'] = cls['cluster']
        # vld = cluster_validator(data, validate_keys=valid_keys + ['size'])
        # print(vld)

        # urban/rural
        urban = train[train.resident == 1]
        score_urban, cls_urban = clustering_modelling(urban, vars=vars, epoch=13, n_clusters=1, display=False)
        data['cluster_urban'] = np.nan
        data.loc[urban.index, 'cluster_urban'] = cls_urban['cluster']

        rural = train[train.resident == 0]
        score_rural, cls_rural = clustering_modelling(rural, vars=vars, epoch=13, n_clusters=1, display=False)
        data['cluster_rural'] = np.nan
        data.loc[rural.index, 'cluster_rural'] = cls_rural['cluster']

        # collect all the K and silhouette scores
        # NB. changed on 2024-02-25, `[score_urban['silhouette_score'].argmax() + 2]` was applied.
        silhouette[f'all-{tag}'] = score['silhouette_score'].tolist() + [score['silhouette_score'].mean()]
        silhouette[f'urban-{tag}'] = score_urban['silhouette_score'].tolist() + [score_urban['silhouette_score'].mean()]
        silhouette[f'rural-{tag}'] = score_rural['silhouette_score'].tolist() + [score_rural['silhouette_score'].mean()]

        # output the final dataset of clustering
        data.to_excel(path / f'cluster-{tag}.xlsx', index=False)

    # reset columns by MultiIndex and remove the last row
    idx = [(k, np.argmax(silhouette[k].values) + 2) for k in silhouette.columns]
    silhouette.columns = pd.MultiIndex.from_tuples(idx)
    silhouette.index = np.arange(2, len(silhouette) + 1).tolist() + ['mean']

    # output with highlight for silhouette scores and its mean values
    stl = silhouette.style.background_gradient(cmap='YlOrRd', axis=None)
    stl.to_excel(path / 'cluster-score.xlsx')
