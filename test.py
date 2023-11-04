# Clustering script
#

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import shap as shap
from IPython.display import display


from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score




if __name__ == '__main__':
    # load data and tests
    from pathlib import Path
    datafile = Path('data') / 'mergedata-1031.csv'
    data = pd.read_csv(datafile)


    # preprocessing
    data = data.fillna(0).replace(-99, 0)
    # convert province/city id
    data['log_en_total'] = np.log(data['en_total'].values + 1)
    data['log_en_total_percap'] = np.log(data['en_total'].values/data['size'].values + 1)
    data['log_expenditure'] = np.log(data['expenditure'].values + 1)
    data['log_raw_income'] = np.log(data['raw_income'].values + 1)
    data['log_income_percap'] = np.log(data['income_percap'].values + 1)

    data['province_id'] = data['province'].apply(lambda x: list(data['province'].unique()).index(x))
    data['prefecture_id'] = data['prefecture'].apply(lambda x: list(data['prefecture'].unique()).index(x))
    # final list of variables
    var_demo = ['prefecture_id', 'region', 'age', 'house_area', 'size']
    var_econ = ['log_raw_income', 'log_expenditure', 'log_income_percap']
    var_live = ['outside', 'live_days']
    var_app = ['num_cooking', 'power_cooking', 'freq_cooking', 'time_cooking',
               'num_water_heater', 'freq_water_heater', 'time_water_heater',
               'label_water_heater', 'num_ac', 'freq_ac', 'power_ac', 'time_ac',
               'label_ac', 'type_heating', 'time_heating', 'area_heating',
               'cost_heating', 'own_vehicle', 'emit_vehicle', 'fuel_vehicle',
               'fuel_price_vehicle', 'cost_vehicle', 'vehicle_num',
               'vehicle_dist', 'vehicle_fuel', 'vehicle_use']
    var_fuels = ['fuel1', 'fuel2', 'fuel3', 'fuel4', 'fuel5',
                 'fuel6', 'fuel7', 'fuel8', 'fuel9', 'fuel10']
    var_family_structure = ['demographicType','IF_single_elderly','IF_singleAE','IF_singleA',
                            'IF_couple_elderly','IF_coupleA','IF_singleWithChildren','IF_coupleWithChildren',
                            'IF_grandparentKids','IF_bigFamily','childrenNumber','elderNumber','IF_existElderly',
                            'IF_existChildren']

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
    vars = var_demo + var_econ + var_app + var_live + var_family_structure
    dep_var = 'log_en_total'
    min_weight = 0.01
    max_iteration = 10000
    alpha_range = np.linspace(0.001, 0.1, 1000)
    min_weight = 0.01

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

    # Use SHAP explain lasso variable importance
    x_display = data[vars].fillna(0).copy(True)
    explainer = shap.LinearExplainer(la, x_display)
    shap_values = explainer(x_display)
    fig = shap.summary_plot(shap_values, x_display,show=False)
    plt.savefig('img/shap1.png', dpi=200)
    plt.clf()
    fig = shap.summary_plot(shap_values, x_display, plot_type="bar",show=False)
    plt.savefig('img/shap2.png', dpi=200)
    plt.clf()

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

    vars = var_after_lasso['vars'].values.tolist()
    index: str = 'id'
    epoch = 10
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

    fig,axes = plt.subplots(epoch-2, 1)
    fig.set_size_inches(10, 12)
    axes_list = []

    for n in tqdm(range(2, epoch, 1)):
        # baseline: K-Means, use n_cluster = 3 as default
        param["n_clusters"] = n
        km = KMeans(**param).fit(train)
        y_pred = km.predict(train)
        eva += [[silhouette_score(train, y_pred), davies_bouldin_score(train, y_pred)]]

        # plot part
        # Create a subplot with 1 row and 2 columns

        n_clusters = int(n)
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        n_plot = n-2
        axes[n_plot].set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        axes[n_plot].set_ylim([0, len(train) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        cluster_labels = y_pred

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(train, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(train, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            axes[n_plot].fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axes[n_plot].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # axes[n_plot].set_title("The silhouette plot for the various clusters.")
        axes[n_plot].set_xlabel("The silhouette coefficient values")
        axes[n_plot].set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        axes[n_plot].axvline(x=silhouette_avg, color="red", linestyle="--")

        axes[n_plot].set_yticks([])  # Clear the yaxis labels / ticks
        axes[n_plot].set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        axes_list.append(axes[n_plot])

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d - %d" % (2,n_clusters)),
                 fontsize=14, fontweight='bold')
    plt.savefig('img/Silhouette1.png', dpi=200)
    plt.show()


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


    len(var_demo+var_econ+var_app+var_live+var_family_structure)
    len(vars)

    output_data = la_coef
    output_path = Path('output_data') / 'lasso_coef-1101.csv'
    output_data.to_csv(output_path, sep=',', index=True, header=True)
