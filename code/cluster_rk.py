# new clustering method for nature energy revision
#

import json
from pathlib import Path

import os as os
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
# set project file path (pls remove .parent when test py.file)
Savefolder_PATH_pic = Path('output_pic') # set output figure folder path
Savefolder_PATH_data = Path('output_data') # set output data folder path
Loadfile_PATH = Path('data/dataForAnalysis1031_aftermerge.csv') # set input data folder path

# choose independent var
def get_timestamp():
    return datetime.now().strftime('%m%d%H%M')

def normalization(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x

def lasso(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, picname):
    x = data[ind_vars].fillna(0).copy(True).values
    y = data[dep_var].copy(True).values
    # normalization process
    for i in range(0, (len(ind_vars) - 1)):
        x[:, i] = normalization(x[:, i])

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
    la_coef["vars"] = ind_vars
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
        fig.savefig(CWD_PATH / Savefolder_PATH_pic / f"variable-importance-figure-{get_timestamp()}-{picname}.png", format="png",
                    dpi=300)
        plt.show()

    var_after_lasso_var = var_after_lasso['vars']
    x_lasso = data[var_after_lasso_var].fillna(0).copy(True).values
    return x_lasso

def cluster(data, epoch, picname):
    train = data.copy()  # could use fitted(post-lasso) or X (raw)
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
        plt.savefig(CWD_PATH/Savefolder_PATH_pic/str(f'cluster-{get_timestamp()}-{picname}.png'), dpi=200)
        plt.show()

    # rerun the clustering model
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(train)
    #y_pred = km.predict(train)
    y_pred = km.labels_
    return y_pred

def main(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, epoch, picname):
    x_lasso = lasso(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, picname)
    y_pred = cluster(x_lasso, epoch, picname)
    # make sure you're passing `cluster` back to the raw dataset
    data["cluster"] = y_pred
    outputdata = data
    readFinalFile = outputdata
    output_filename = f"dataForAnalysis1031-{get_timestamp()}-{picname}.csv"
    output_path = CWD_PATH / Savefolder_PATH_data / output_filename
    readFinalFile.to_csv(output_path, sep=',', index=True, header=True)


if __name__ == '__main__':
    file_root = CWD_PATH / Loadfile_PATH
    missing_values = [999999, -99]  # add missing value type
    read_dataForAnalysis = pd.read_csv(file_root, encoding='utf8', na_values=missing_values)
    inputdata_first_filter = read_dataForAnalysis
    # add new variable
    inputdata_first_filter['percapita_en_total_no_vehicle'] = inputdata_first_filter['en_total_no_vehicle']/inputdata_first_filter['size']

    # exclude useless var
    select_exclude_name = [
        'family_size',
        'income_percapita',
        'expenditure_percapita',
        # 'ages_mean',
        # 'gender_codes',
        'income_list',
        'expenditure_list',
        # 'NumCookingEquip_list',
        # 'NumWashMach_list',
        # 'NumDryer_list',
        # 'NumTV_list',
        # 'NumPC_list',
        # 'NumFluLamp_codes',
        # 'NumIcdLamp_codes',
        # 'NumESLamp_codes',
        # 'NumHeater_list',
        # 'NumWaterHeater_list',
        # 'NumAC_list',
        # 'NumVehicle_codes',
        # 'HouseAge_codes',
        # 'WindowFrame_codes',
        # 'GlassType_codes',
        # 'Reno_WindowGlass_codes',
        # 'Reno_WindowGap_codes',
        # 'Reno_Wall_codes',
        # 'Reno_OtherHeatInsu_codes',
        # 'WinterSunshine_codes',
        # 'SummerSunshine_codes',
        # 'Weektime_list',
        # 'Cookingtime_codes',
        # 'Fridgetime_codes',
        # 'Washmachinetime_codes',
        # 'TVtime_codes',
        # 'PCtime_codes',
        # 'FluLamptime_codes',
        # 'IcdLamptime_codes',
        # 'ESLamptime_codes',
        # 'MthHeattime_codes',
        # 'DayHeatertime_codes',
        # 'Waterheatingtime_codes',
        # 'ACtime_codes',
        'gender',
        'age',
        'house_area',
        'media',
        'outside',
        'aware_trust',
        'aware_harm',
        'aware_justice',
        'aware_happy',
        'num_cooking',
        'power_cooking',
        'freq_cooking',
        'time_cooking',
        'num_water_heater',
        'freq_water_heater',
        'time_water_heater',
        'label_water_heater',
        'num_ac',
        'freq_ac',
        'power_ac',
        'time_ac',
        'label_ac',
        'type_heating',
        'time_heating',
        'area_heating',
        'cost_heating',
        'own_vehicle',
        'emit_vehicle',
        'fuel_vehicle',
        'fuel_price_vehicle',
        'cost_vehicle',
        'vehicle_num',
        'vehicle_dist',
        'vehicle_fuel',
        'vehicle_use'
    ]
    columns_name = list(inputdata_first_filter.columns.values)
    var_name = list(set(columns_name) - set(select_exclude_name))
    inputdata_first_filter = pd.DataFrame(inputdata_first_filter[var_name])

    # missing data cleaning
    missing_series = inputdata_first_filter.isnull().sum() / inputdata_first_filter.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index': 'col',
                                            0: 'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct', ascending=False).reset_index(drop=True)
    missing_df_filtered = missing_df[missing_df['missing_pct'] < 0.10]
    select_var_name = list(missing_df_filtered['col'])

    VAR_LABELS = dict()
    for i in range(0, (len(columns_name) - 1)):
        VAR_LABELS[columns_name[i]] = columns_name[i]

    # add exclude var name
    select_exclude_name = [
        'id',
        'region_codes',
        'Unnamed: 0',
        'Unnamed: 0.1',
        'en_ac',
        'en_computer',
        'en_cooking',
        'en_freezing',
        'en_heating',
        'en_laundry',
        'en_lighting',
        'en_television',
        'en_vehicle',
        'en_waterheating',
        'percapita_en_ac',
        'percapita_en_computer',
        'percapita_en_cooking',
        'percapita_en_freezing',
        'percapita_en_heating',
        'percapita_en_laundry',
        'percapita_en_lighting',
        'percapita_en_television',
        'percapita_en_vehicle',
        'percapita_en_waterheating',
        'TotalHouseholdEnergyUse',
        'en_total',
        'en_total_no_vehicle',
        'fuel1',
        'fuel2',
        'fuel3',
        'fuel4',
        'fuel5',
        'fuel6',
        'fuel7',
        'fuel8',
        'fuel9',
        'fuel10',
        'percapita_HouseholdEnergyUse'
    ]
    var_name = list(set(select_var_name) - set(select_exclude_name))

    inputdata = pd.DataFrame(inputdata_first_filter[var_name])
    inputdata_no_na = inputdata.dropna(axis=0, how='any')
    dependent_var = ['percapita_en_total_no_vehicle']
    independent_var = list(set(var_name) - set(dependent_var))
    independent_var = list(set(independent_var) - set(['resident_codes','resident']))  # delete group code
    type(inputdata_no_na[dependent_var])
    index = inputdata_no_na[dependent_var] > 0
    inputdata_no_na = inputdata_no_na[index.squeeze()]
    # extreme value cleaning
    filter_data = inputdata_no_na
    n_iqr = 3
    # pls check the consistency between select_exclude_name and extreme_check_var
    # pls make sure all variable is exist
    extreme_check_var = [
        'raw_income',
        'income_percap',
        'percapita_en_total_no_vehicle']

    for i in range(0, (len(extreme_check_var) - 1)):
        name_temp = extreme_check_var[i]
        data_temp = filter_data[name_temp]

        upper_quartile = np.percentile(data_temp, 75)
        lower_quartile = np.percentile(data_temp, 25)

        iqr = upper_quartile - lower_quartile
        upper_whisker = data_temp[data_temp <= upper_quartile + n_iqr * iqr].max()
        lower_whisker = data_temp[data_temp >= lower_quartile - n_iqr * iqr].min()

        filter_data = filter_data[(filter_data[name_temp] < upper_whisker)]
    data_urban = filter_data[(filter_data['resident_codes'] == 0)]  # change urban and rural (0:urabn, 1:rural)
    data_rural = filter_data[(filter_data['resident_codes'] == 1)]

    ind_vars = independent_var
    dep_var = dependent_var
    min_weight = 0.05
    alpha_range = np.linspace(0.001, 0.1, 1000)
    max_iteration = 10e3
    epoch = 10

    data = filter_data
    picname = 'all'
    main(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, epoch, picname)

    data = data_urban
    picname = 'urban'
    main(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, epoch, picname)

    data = data_rural
    picname = 'rural'
    main(data, ind_vars, dep_var, min_weight, alpha_range, max_iteration, epoch, picname)
    # make sure you're passing `cluster` back to the raw dataset
