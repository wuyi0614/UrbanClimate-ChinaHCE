# Clustering script
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap as shap

from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def lasso_modelling(data: pd.DataFrame,
                    vars: list,
                    dep_var: str,
                    min_weight=None,
                    alpha_range: list = None,
                    max_iteration=1000,
                    display=True):
    if alpha_range is None:
        alpha_range = np.linspace(0.001, 0.1, 1000)

    if min_weight is None:
        min_weight = 0.05

    x = data[vars].fillna(0).copy(True).values
    y = data[dep_var].copy(True).values

    model = Lasso()
    result = GridSearchCV(model,
                          param_grid={'alpha': alpha_range, 'max_iter': [max_iteration]},
                          cv=5,
                          scoring='neg_mean_absolute_error')
    result.fit(x, y)
    print('MAE: %.5f' % result.best_score_)
    print('Optimal param：\n', result.best_params_)
    alpha = result.best_params_

    # with optimal model
    la = Lasso(**alpha).fit(x, y)  # ... find the best alpha
    la_coef = pd.DataFrame(la.coef_, columns=["coef"])
    la_coef["vars"] = vars
    la_coef = la_coef.sort_values("coef", ascending=False)
    la_coef["colors"] = "#639DBC"
    la_coef.loc[la_coef.coef < 0, "colors"] = "#B6C438"
    var_after_lasso = la_coef[la_coef.coef.abs() >= min_weight]
    print(f"{len(var_after_lasso)} variables are filtered with weight={min_weight}")

    # output distribution of weights of variables
    if display:
        fig = plt.figure(figsize=(8, 10), dpi=120)
        x_range = range(len(la_coef))
        plt.barh(x_range, la_coef.coef.values, color=la_coef.colors.values, label=f"Lasso alpha={min_weight}")

        ticks = [VAR_LABELS[k] for k in la_coef.vars.values]
        plt.yticks(x_range, labels=ticks, size=9, rotation=0)
        plt.ylabel("Household features")
        plt.xlabel("Feature importance")

        plt.margins(0.01)
        plt.tight_layout()
        plt.grid(axis="x", color="grey", alpha=0.3)
        fig.savefig('img/lasso1.png', format='png', dpi=300)
        plt.show()

    return var_after_lasso


def clustering_modelling(data: pd.DataFrame,
                         vars: list,
                         index: str = 'id',
                         epoch=10,
                         display=True):
    """Modelling process for clustering"""
    # reconcile with features
    x = data[vars].fillna(0).values
    train = x.copy()  # could use fitted(post-lasso) or X (raw)

    # clustering pipeline
    param = {
        "n_clusters": 1,
        "init": "k-means++",
        "algorithm": "elkan",
        "random_state": 0
    }

    eva = []
    for n in tqdm(range(2, epoch, 1)):
        # baseline: K-Means, use n_cluster = 3 as default
        param["n_clusters"] = n
        km = KMeans(**param).fit(train)
        y_pred = km.predict(train)
        eva += [[silhouette_score(train, y_pred), davies_bouldin_score(train, y_pred)]]

    exp = pd.DataFrame(eva, columns=["silhouette_score", "calinski_harabasz_score"])
    print(exp)

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
    print(f"Counted households: {counted}")

    # check the averaged percap emissions
    counted = data[["en_total", 'cluster']].groupby('cluster').mean()
    print(f"Classified energy consumption: {counted}")

    # output
    return data.copy(True)


if __name__ == '__main__':
    # load data and tests
    from pathlib import Path
    datafile = Path('data') / 'mergedata-1104.csv'
    data = pd.read_csv(datafile)


    # preprocessing
    data = data.fillna(0).replace(-99, 0)
    # convert province/city id
    data['log_en_total'] = np.log(data['en_total'].values + 1)
    data['log_en_total_percap'] = np.log(data['en_total'].values/data['size'].values + 1)
    data['log_expenditure'] = np.log(data['expenditure'].values + 1)
    data['log_raw_income'] = np.log(data['raw_income'].values + 1)
    data['log_income_percap'] = np.log(data['income_percap'].values + 1)
    data['log_en_total_no_vehicle_percap'] = np.log(data['en_total_no_vehicle_percap'].values + 1)

    data['province_id'] = data['province'].apply(lambda x: list(data['province'].unique()).index(x))
    data['prefecture_id'] = data['prefecture'].apply(lambda x: list(data['prefecture'].unique()).index(x))
    # final list of variables
    var_demo = ['prefecture_id', 'region', 'age', 'house_area', 'size','region_codes']
    var_econ = ['log_raw_income', 'log_expenditure', 'log_income_percap']
    var_live = ['outside', 'live_days']
    var_app = ['num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
               'num_water_heater', 'freq_water_heater', 'time_water_heater',
               'label_water_heater', 'num_ac', 'freq_ac', 'power_ac', 'time_ac',
               'label_ac', 'type_heating', 'time_heating', 'area_heating',
               'cost_heating',
               'own_vehicle', 'emit_vehicle', 'fuel_vehicle',
               'fuel_price_vehicle', 'cost_vehicle', 'vehicle_num',
               'vehicle_dist', 'vehicle_fuel', 'vehicle_use'
               ]
    var_fuels = ['fuel1', 'fuel2', 'fuel3', 'fuel4', 'fuel5',
                 'fuel6', 'fuel7', 'fuel8', 'fuel9', 'fuel10']
    var_family_structure = ['demographicType','IF_single_elderly','IF_singleAE',
                            'IF_couple_elderly','IF_coupleA','IF_coupleWithChildren',
                            'childrenNumber','elderNumber','IF_existElderly',
                            'IF_existChildren',
                            'IF_singleA',
                            'IF_grandparentKids',
                            'IF_bigFamily',
                            'IF_singleWithChildren'
                            ]

    """ Cache for the best option of clustering
    var_lasso = lasso_modelling(data, vars=var_demo+var_econ+var_app+var_live,
                                dep_var='log_en_total', min_weight=0.01, max_iteration=10000)

    vars = ['live_days', 'num_water_heater', 'vehicle_dist', 'freq_cooking', 'time_heating', 'region', 'vehicle_use', 
    'num_cooking', 'house_area', 'freq_water_heater', 'label_water_heater', 'freq_ac', 'size', 'vehicle_fuel', 
    'log_expenditure', 'log_raw_income', 'num_ac', 'type_heating']
    """
    # Generate Y label name for lasso
    columns_name = list(data.columns.values)
    VAR_LABELS = dict()
    for i in range(0, (len(columns_name))):
        VAR_LABELS[columns_name[i]] = columns_name[i]

    # feature engineering
    var_lasso = lasso_modelling(data, vars=var_demo+var_app+var_live+var_family_structure,
                                dep_var='log_en_total', min_weight=0.01, max_iteration=10000)

    # clustering test
    vars = var_lasso['vars'].values.tolist()
    output_data = clustering_modelling(data, vars=vars)
    output_path = Path('output_data') / 'cluster_output-1104.csv'
    output_data.to_csv(output_path, sep=',', index=True, header=True)
    # # urban/rural
    # urban = data[data.resident == 1]
    # rural = data[data.resident == 0]
    # clustering_modelling(urban, vars=vars)
    # clustering_modelling(rural, vars=vars)

    view = output_data[vars+['cluster']].groupby('cluster').mean()
    output_data = view
    output_path = Path('output_data') / 'classification_metrics-1104.csv'
    output_data.to_csv(output_path, sep=',', index=True, header=True)