# Import libraries

# For data manipulation
import numpy as np
import pandas as pd

# For standardizing features. We'll use the StandardScaler module.
from sklearn.preprocessing import StandardScaler

# For compute the z score and drop outliers
from scipy import stats

# For optimization number of clusters
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.style import set_palette

# Clustering method
from sklearn.cluster import KMeans

# For graphic plot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

def analysis(df,colunas_caracteristicas,columns_order, columns_behavior, classes_name):
    # Sorting the classes hierarchically
    df_new = df.groupby('clusters')[columns_order].mean().reset_index()
    df_new = df_new.sort_values(by=columns_order,ascending=columns_behavior).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[['clusters','index']], on='clusters')
    df_final = df_final.drop(['clusters'],axis=1)
    df_final = df_final.rename(columns={"index":'clusters'})
    
    if classes_name is not None:
        return classes(df_final, colunas_caracteristicas, classes_name)
        
    return results(df_final,colunas_caracteristicas)
 
def classes(df,colunas_caracteristicas, classes_name):
    # Add the segment labels
    df['clusters'] = df['clusters'].astype(int)
    classe = 0
    for name in classes_name:
        df.loc[df['clusters'] == classe,'clusters'] = name 
        classe += 1
    print(classes_name)
    return results(df,colunas_caracteristicas)


def results(df_final, colunas_caracteristicas):
    
    # Class interpretation    
    df_model_hierarchy_statistics = df_final.groupby('clusters').agg({col: 'count' if col == "id" else ['max','min', 'mean'] for col in df_final.drop('clusters', axis=1).columns})
    df_model_hierarchy_statistics.reset_index(inplace=True)
    
    # Heatmap of features and classes
    plot_model_hierarchy_features_heatmap = plot_features_heatmap(df_final,colunas_caracteristicas)


    # Matrix Scater of features and classes
    plot_model_hierarchy_features_scatter = plot_features_scatter_plotly(df_final)
    
    return {
        "model_hierarchy_raw": df_final,
        "model_hierarchy_statistics_raw": df_model_hierarchy_statistics,
        "plot_model_hierarchy_features_heatmap": plot_model_hierarchy_features_heatmap,
        "plot_model_hierarchy_features_scatter": plot_model_hierarchy_features_scatter,
  
    }
    

def main(
    df,
    coluna_id,
    colunas_caracteristicas,
    control_number_clusters,
    number_clusters,
):
    
    """
    Classifies customers in clusters by K-means method
    based on features passed on the dataset
    ----------
    df : Pandas DataFrame
        Input dataset
    coluna_data : str
        Client ID column name
    colunas_regressores : list<str>
        List of all the features column names
    
    Returns
    -------
    JSON with predicted clusters, cluster statistics and plot figures
    """
    
    # Data processing
    #colunas_caracteristicas = [x for x in colunas_caracteristicas.split(",") if x]
    
    df = df[[coluna_id] + colunas_caracteristicas] 
    

    # Standardizing data, so that all features have equal weight
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df.drop(coluna_id, axis=1))
    df_std = pd.DataFrame(data = df_std, columns = colunas_caracteristicas)
    
    # Remove outliers by z-score limit 3
    outlier_mask = (np.abs(stats.zscore(df_std)) < 3).all(axis=1)
    df_std =df_std[outlier_mask]
    df_final_clients = df[outlier_mask]
    
    # Save dropped ID's
    df_drop_clients = df[outlier_mask.apply(lambda x: not x)]


    #KElbowVisualizer implements the “elbow” method to select the optimal number of clusters by fitting the model with a range of values for K
    plt.style.use('dark_background')
    plot_optimal_clusters_k, ax = plt.subplots(figsize=(6,4))
    set_palette('sns_pastel')
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2,15), metric='distortion',timings=False,ax=ax)
    visualizer.vline_color = '#F4F4F4'
    visualizer.fit(df_std)
    visualizer.show()
    k_clusters = visualizer.elbow_value_ or 2 # The attribute elbow_value_ return the optimal value of k
    
    
    
    # We run K-means with a fixed number of clusters.
    if control_number_clusters:
        k_clusters = number_clusters
    kmeans = KMeans(n_clusters = k_clusters, init = 'k-means++', random_state = 42)
    kmeans.fit(df_std)
    
    # Clusters results
    df_final_clients['clusters'] = kmeans.labels_
    
    # Class interpretation
    df_clusters_statistics_raw = df_final_clients.groupby('clusters').agg({col: 'count' if col == coluna_id else ['max','min', 'mean'] for col in df_final_clients.drop('clusters', axis=1).columns})
    df_clusters_statistics_raw.reset_index(inplace=True)
    
    # Heatmap of features and classes
    plot_clusters_features_heatmap_raw = plot_features_heatmap(df_final_clients,colunas_caracteristicas)


    # Matrix Scater of features and classes
    plot_clusters_features_scatter_raw = plot_features_scatter_plotly(df_final_clients)


    

    return {
        "clustering_raw": df_final_clients,
        "clusters_statistics_raw": df_clusters_statistics_raw,
        "plot_optimal_clusters_k": plot_optimal_clusters_k,
        "plot_clusters_features_heatmap_raw": plot_clusters_features_heatmap_raw,
        "plot_clusters_features_scatter_raw": plot_clusters_features_scatter_raw,
        "dropped_ids_raw": df_drop_clients
  
    }

    
def plot_features_heatmap(df, columns):
    plt.style.use("dark_background")
    # Heatmap of features and classes
    fig, ax = plt.subplots(figsize=(6,4))
    df_heatmap = df.groupby('clusters')[columns].mean()
    df_norm_col=(df_heatmap  - df_heatmap.mean())/df_heatmap.std()
    sns.heatmap(df_norm_col, cmap='viridis', ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([df_norm_col.values.min(), df_norm_col.values.max()])
    cbar.set_ticklabels(['Low\nAvg', 'High\nAvg'])
    return fig


def plot_features_scatter_seaborn(df):
    plt.style.use("dark_background")
    fig = sns.pairplot(df.drop(['id'], axis=1), hue="clusters",
                palette="Set2")
    return fig

def plot_features_scatter_plotly(df):
    df['clusters'] = df['clusters'].astype(str)
    fig = ff.create_scatterplotmatrix(df.drop(['id'], axis=1), diag='histogram',index='clusters',
                                  height=680, width=680,colormap= ["#3C3F48","#00C0D9","#C0143C","#E4EAF1","#FD4239","#979BA3"], title = None, 
                                 size = 5)
    fig.update_layout(template='plotly_dark')

    return fig



