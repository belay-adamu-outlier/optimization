import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt

def run_clustering_analysis():
    """Run the clustering analysis on optimization results"""
    try:
        sim_data = pd.read_csv('optimization_results_20250715_164420.csv', skiprows=11)
    except FileNotFoundError:
        print("Warning: optimization_results_20250715_164420.csv not found. Clustering analysis requires this file.")
        return

    # filter out rows where TEDI is greater than 15
    sim_data = sim_data[sim_data.iloc[:,-3] <= 15]

    # Load window data
    try:
        with open('windows.json', 'r') as f:
            window_data = json.load(f)
    except FileNotFoundError:
        print("Warning: windows.json not found. Clustering analysis requires this file.")
        return

    # Insert U-Value and SHGC columns
    u_values = [window_data[window]['U-Value'] for window in sim_data.iloc[:,4]]
    shgcs = [window_data[window]['SHGC'] for window in sim_data.iloc[:,4]]

    sim_data.insert(5, 'U-Value', u_values)
    sim_data.insert(6, 'SHGC', shgcs)

    # Drop unnecessary columns
    window_column = sim_data.iloc[:,4]
    sim_data.drop(columns=['window', 'Generation', 'Iteration'], inplace=True)

    # Change R values to floats
    for i in range(len(sim_data)):
        sim_data.iloc[i,0] = float(sim_data.iloc[i,0].replace(' - Wall', '').replace('R', ''))
        sim_data.iloc[i,1] = float(sim_data.iloc[i,1].replace(' - Roof', '').replace('R', ''))

    # #encode the last column
    pareto_column = sim_data.iloc[:,-1]
    label_encoder = LabelEncoder()
    transformed_pareto_column = label_encoder.fit_transform(pareto_column)
    sim_data.iloc[:,-1] = transformed_pareto_column

    # Scale the data
    standard_scaler = StandardScaler()
    sim_data.iloc[:,:] = standard_scaler.fit_transform(sim_data.iloc[:,:])
    # sim_data.iloc[:,-1] = standard_scaler.fit_transform(sim_data.iloc[:,-1])

    # cluster the data

    # intertias = []

    # for i in range(1, 30):
    #     kmeans = KMeans(n_clusters=i, random_state=0)
    #     kmeans.fit(sim_data.iloc[:,:])
    #     intertias.append(kmeans.inertia_)

    # plt.plot(range(1, 30), intertias, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method')
    # plt.show()

    kmeans = KMeans(n_clusters=20, random_state=0)
    kmeans.fit(sim_data.iloc[:,:])

    # Plot the data
    plt.scatter(sim_data.iloc[:,-3], sim_data.iloc[:,-2], c=kmeans.labels_)
    plt.title('Clustered Solutions')
    plt.xlabel('ASHRAE Overheating Hours (Standardized)')
    plt.ylabel('Mechanical Energy Usage Intensity (Standardized)')
    plt.show()

    # # Revert the scaling
    # sim_data.iloc[:,:] = standard_scaler.inverse_transform(sim_data.iloc[:,:])

    # # remove the U-Value and SHGC columns and reintroduce the window column
    # sim_data.drop(columns=['U-Value', 'SHGC'], inplace=True)
    # sim_data.insert(4, 'window', window_column)

    # # Append the lables to sim_data
    # sim_data['cluster'] = kmeans.labels_

    # print(sim_data.columns)
    # sim_data.to_csv('clustered_data.csv', index=False)

if __name__ == "__main__":
    run_clustering_analysis()








