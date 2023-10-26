# new clustering method for nature energy revision
#

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from holoviews.ipython import display
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


# default saving filepath
## Pls change all the filepath
CWD_PATH = Path.cwd().parent
#CWD_PATH = Path.cwd() /'data'
# set project file path (pls remove .parent when test py.file)
Savefolder_PATH_pic = Path('output_pic') # set output figure folder path
Savefolder_PATH_data = Path('data') # set output data folder path
Loadfile_PATH = Path('data/dataForAnalysis1025_aftermerge.csv') # set input data folder path

# choose independent var
def get_timestamp():
    return datetime.now().strftime('%m%d%H%M')

def normalization(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x

if __name__ == '__main__':
    file_root = CWD_PATH/Loadfile_PATH
    # missing data cleaning
    missing_values = [999999] # add missing value type
    read_dataForAnalysis = pd.read_csv(file_root,encoding='utf8',na_values = missing_values)

    missing_series = read_dataForAnalysis.isnull().sum() / read_dataForAnalysis.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index': 'col',
                                            0: 'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)
    missing_df_filtered = missing_df[missing_df['missing_pct'] < 0.10]
    select_var_name = list(missing_df_filtered['col'])

    VAR_LABELS = dict()
    for i in range(0, (len(select_var_name) - 1)):
        VAR_LABELS[select_var_name[i]] = select_var_name[i]

    # add exclude var name
    select_exclude_name = [
        'id',
        'resident_codes',
        'region_codes',
        'Unnamed: 0',
        'Unnamed: 0.1',
        'ac_emission',
        'computer_emission',
        'cooking_emission',
        'freezing_emission',
        'heating_emission',
        'laundry_emission',
        'lighting_emission',
        'television_emission',
        'vehicle_emission',
        'waterheating_emission',
        'percapita_ac_emission',
        'percapita_computer_emission',
        'percapita_cooking_emission',
        'percapita_freezing_emission',
        'percapita_heating_emission',
        'percapita_laundry_emission',
        'percapita_lighting_emission',
        'percapita_television_emission',
        'percapita_vehicle_emission',
        'percapita_waterheating_emission',
        'TotalHouseholdEmission']
    var_name = list(set(select_var_name) - set(select_exclude_name))

    inputdata = pd.DataFrame(read_dataForAnalysis[var_name])
    inputdata_no_na = inputdata.dropna(axis=0, how='any')
    dependent_var = ['percapita_HouseholdEmission']
    independent_var = list(set(var_name) - set(dependent_var))

    # extreme value cleaning
    filter_data = inputdata_no_na
    n_iqr = 3
    extreme_check_var = [
        'income_list',
        'expenditure_list',
        'income_percapita',
        'expenditure_percapita',
        'percapita_HouseholdEmission']

    for i in range(0, (len(extreme_check_var)-1)):
        name_temp = extreme_check_var[i]
        data_temp = filter_data[name_temp]

        upper_quartile = np.percentile(data_temp, 75)
        lower_quartile = np.percentile(data_temp, 25)

        iqr = upper_quartile - lower_quartile
        upper_whisker = data_temp[data_temp <= upper_quartile + n_iqr * iqr].max()
        lower_whisker = data_temp[data_temp >= lower_quartile - n_iqr * iqr].min()

        filter_data = filter_data[(filter_data[name_temp] < upper_whisker)]

    data = filter_data
    vars = independent_var
    dep_var = dependent_var
    min_weight = 0.05
    alpha_range = np.linspace(0.001, 0.1, 1000)
    max_iteration=10e3
    x = data[vars].fillna(0).copy(True).values
    y = data[dep_var].copy(True).values
    # normalization process
    for i in range(0, (len(vars)-1)):
        x[:,i] = normalization(x[:,i])

    y = normalization(y)

    model = Lasso()
    result = GridSearchCV(model, param_grid={'alpha': alpha_range, 'max_iter': [max_iteration]}, cv=5,
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
        plt.barh(x_range, la_coef.coef.values, color=la_coef.colors.values, label="Lasso alpha=0.04")

        ticks = [VAR_LABELS[k] for k in la_coef.vars.values]
        plt.yticks(x_range, labels=ticks, size=9, rotation=0)
        plt.ylabel("Household features")
        plt.xlabel("Feature importance")

        plt.margins(0.01)
        plt.tight_layout()
        plt.grid(axis="x", color="grey", alpha=0.3)
        fig.savefig(CWD_PATH/Savefolder_PATH_pic/ f"variable-importance-figure-{get_timestamp()}.png", format="png", dpi=300)
        plt.show()

    var_after_lasso_var = var_after_lasso['vars']
    x_lasso = data[var_after_lasso_var].fillna(0).copy(True).values


    train = x_lasso.copy()  # could use fitted(post-lasso) or X (raw)
    epoch = 10
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
        y_pred = km.labels_
        #y_pred = km.predict(train) # check km.predict vs km.labels_
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

        plt.axvspan(n_clusters-2.5, n_clusters-1.5, 0, 0.975,
                    facecolor="none", edgecolor="red", linewidth=2, linestyle="--")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette score")  # the higher, the better
        plt.grid(axis="y", color="grey", alpha=0.3)
        plt.savefig(CWD_PATH/Savefolder_PATH_pic/str(f'cluster-{get_timestamp()}.png'), dpi=200)
        plt.show()

    # rerun the clustering model
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(train)
    #y_pred = km.predict(train)
    y_pred = km.labels_

    # make sure you're passing `cluster` back to the raw dataset
    data["cluster"] = y_pred

    # need to revise below
    #     # statistics
    #     counted = data[["KEY", "cluster"]].groupby("cluster").count()
    #     print(f"Counted households: {counted}")
    #
    #     # check the averaged percap emissions
    #     counted = data[["emits_per", "cluster"]].groupby("cluster").mean()
    #     print(f"Counted emission percap: {counted}")

    readFinalFile = data
    output_filename = 'dataForAnalysis1026.csv'
    output_path = CWD_PATH / Savefolder_PATH_data / output_filename
    readFinalFile.to_csv(output_path, sep=',', index=True, header=True)

    data_cluster_0 = data[(data['cluster'] == 0)] # select cluster 0

    readFinalFile = data_cluster_0
    output_filename = 'dataForAnalysis1026_cluster_0.csv'
    output_path = CWD_PATH / Savefolder_PATH_data / output_filename
    readFinalFile.to_csv(output_path, sep=',', index=True, header=True)

