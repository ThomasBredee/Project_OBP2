#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester                          #
#                                                       #
#########################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CandidateRanking:

    def __init__(self):
        pass

    # Define the function to remove everything after '_' in column names and index
    @staticmethod
    def _clean_column_and_index_labels(matrix):
        matrix_copy = matrix.copy()  # Create a copy to avoid modifying the original
        matrix_copy.columns = [col.split('_')[0] for col in matrix_copy.columns]
        matrix_copy.index = [idx.split('_')[0] for idx in matrix_copy.index]
        return matrix_copy

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1)** 2 + (lon2 - lon1)** 2)

    def greedy(self, matrix):
        # Clean column names and index
        matrix_new = self._clean_column_and_index_labels(matrix)

        company_name = matrix_new.index[0]
        grouped_df = matrix_new.groupby(matrix_new.columns, axis=1).sum()
        grouped_df = grouped_df.drop(columns=company_name)
        ranking = grouped_df.sum(axis=0).sort_values(ascending=True).to_frame()
        ranking.columns = ['Total Distance']
        ranking = round(ranking, 0)
        #print(ranking)
        return ranking

    def bounding_box(self, df, matrix):
        matrix = self._clean_column_and_index_labels(matrix)
        company_name = matrix.index[0]
        partner_names = matrix.columns.unique().drop(company_name)
        company_df = df[df['name'] == company_name]
        partner_df = df[df['name'] != company_name]

        middlepoint = (company_df['lat'].mean(), company_df['lon'].mean())
        print(middlepoint)
        distances = [self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon']) for _, row in
                     company_df.iterrows()]
        radius = np.mean(distances)

        # Output the results
        print(f"Average Euclidean distance from the midpoint: {radius:.2f}")

        # circle = plt.Circle((middlepoint[0], middlepoint[1]),  radius, color='r', fill=False)
        fig, ax = plt.subplots()

        circle = plt.Circle((middlepoint[0], middlepoint[1]), radius, color='r', fill=False, linestyle='--',
                            label='Circle with radius')

        count_inside = pd.DataFrame({
            'Partner': partner_names,
            'Percentage': [0] * len(partner_names)  # All zeros in 'percentage' column
        })
        for i in range(len(partner_names)):
            df_temp = partner_df[partner_df['name'] == partner_names[i]]

            plt.figure(figsize=(12, 8))

            ax = plt.gca()
            circle = plt.Circle((middlepoint[1], middlepoint[0]), radius, color='r', fill=False, linestyle='--',
                                label='Circle with radius')

            ax.add_patch(circle)
            plt.scatter(middlepoint[1], middlepoint[0], color='red', label='Midpoint')
            plt.scatter(company_df['lon'], company_df['lat'], color='blue', label='Company Locations')
            df_temp = partner_df[partner_df['name'] == partner_names[i]]

            plt.scatter(df_temp['lon'], df_temp['lat'], color='green', label=f'{partner_names[i]} Locations')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Map for {partner_names[i]} with Circle')
            ax.legend()
            plt.show()

            count = 0
            j = 0
            for _, row in df_temp.iterrows():
                distance = self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon'])
                j += 1
                if distance <= radius:
                    count += 1
            count_inside.loc[i, 'Percentage'] = count / j
        count_inside = count_inside.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        print(count_inside)

    def k_means(self, df,matrix):
        # print(matrix)
        chosen_company = matrix.index[0]
        minimal_clusters = len(df[df['name'] == chosen_company])
        print('m',minimal_clusters)
        partner_names = matrix.columns.unique().drop(chosen_company)
        print(partner_names)
        scaler = StandardScaler()
        df[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df[['lat', 'lon']])

        sil_scores = []
        for k in range(minimal_clusters, minimal_clusters+10):  # Test for 2 to 10 clusters
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df[['lat_scaled', 'lon_scaled']])  # Replace with your data columns
            sil_scores.append(silhouette_score(df[['lat_scaled', 'lon_scaled']], kmeans.labels_))


        print('s',sil_scores)
        print(np.max(sil_scores))
        print("test", np.argmax(sil_scores))

        k = minimal_clusters + np.argmax(sil_scores)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['lat_scaled', 'lon_scaled']])
        print(df)
        self.plot_clusters(df,k)

        clusters_company = df[df['name'] == chosen_company]['cluster'].unique()

        count_inside = pd.DataFrame({
            'Partner': partner_names,
            'Percentage': [0] * len(partner_names)  # All zeros in 'percentage' column
        })
        percentage = []
        for i in partner_names:
            df_temp = df[df['name'] == i]
            df_temp2 = df_temp['cluster'].isin(clusters_company)

            percentage.append(sum(df_temp2)/len(df_temp))
        count_inside['Percentage'] = percentage
        ranking = count_inside.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        print(ranking)
        print(minimal_clusters)
        return ranking

    def plot_clusters(self,df,k):
        plt.figure(figsize=(8, 6))

        # Plot each cluster with a different color
        for cluster in range(k):
            cluster_data = df[df['cluster'] == cluster]
            plt.scatter(cluster_data['lat'], cluster_data['lon'], label=f'Cluster {cluster + 1}')



        plt.title('K-Means Clustering of Locations')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()