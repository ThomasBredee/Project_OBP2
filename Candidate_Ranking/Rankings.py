#########################################################
#                                                       #
# Created on: 15/01/2025                                #
# Created by: Lukas and Wester                          #
#                                                       #
#########################################################


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, euclidean_distances
from sklearn.cluster import DBSCAN
from Algorithms.distance_calculator import RoadDistanceCalculator
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from math import radians


class CandidateRanking:

    def _init_(self):
        self.correlation_df = pd.DataFrame()
        pass

    # Define the function to remove everything after '_' in column names and index
    @staticmethod
    def _clean_column_and_index_labels(matrix):
        matrix_copy = matrix.copy()  # Create a copy to avoid modifying the original
        matrix_copy.columns = [col.split('_')[0] for col in matrix_copy.columns]
        matrix_copy.index = [idx.split('_')[0] for idx in matrix_copy.index]
        return matrix_copy



    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)

    def greedy(self, matrix,comparing):
        # Clean column names and index
        matrix_new = self._clean_column_and_index_labels(matrix)
        company_name = matrix_new.index[0]
        grouped_df = matrix_new.groupby(matrix_new.columns, axis=1).sum()
        grouped_df = grouped_df.drop(columns=company_name)
        ranking = grouped_df.sum(axis=0).sort_values(ascending=True).to_frame()
        ranking.columns = ['Total Distance']
        ranking = round(ranking, 0)
        if comparing == True:
            ranking['Ranking'] = range(1, len(ranking) +1)

        print(ranking)
        return ranking

    def bounding_box(self, df, matrix, comparing):
        matrix = self._clean_column_and_index_labels(matrix)
        company_name = matrix.index[0]
        partner_names = matrix.columns.unique().drop(company_name)
        company_df = df[df['name'] == company_name]
        partner_df = df[df['name'] != company_name]
        middlepoint = (company_df['lat'].mean(), company_df['lon'].mean())
        distances = company_df.apply(
            lambda row: self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon']), axis=1)

        # Radius is the mean distance
        radius = distances.mean()

        """
        if comparing == False:
            fig, ax = plt.subplots()
            circle = plt.Circle((middlepoint[1], middlepoint[0]), radius, color='r', fill=False, linestyle='--',
                                label='Circle with radius')
            ax.add_patch(circle)
            plt.scatter(company_df['lon'], company_df['lat'], color='blue', label='Company Locations')
            plt.scatter(middlepoint[1], middlepoint[0], color='red', label='Midpoint')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title(f'Map of {company_name} Deliveries with Bounding Circle')
            ax.legend()
            plt.show()
        """

        count_inside = pd.DataFrame({
            'Partner': partner_names,
            'Percentage': [0.0] * len(partner_names)
        })

        for i in range(len(partner_names)):
            df_temp = partner_df[partner_df['name'] == partner_names[i]]
            """if comparing == False:
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                circle = plt.Circle((middlepoint[1], middlepoint[0]), radius, color='r', fill=False, linestyle='--',
                                    label='Circle with radius')
                ax.add_patch(circle)
                plt.scatter(middlepoint[1], middlepoint[0], color='red', label='Midpoint')
                plt.scatter(company_df['lon'], company_df['lat'], color='blue', label='Company Locations')
                plt.scatter(df_temp['lon'], df_temp['lat'], color='green', label=f'{partner_names[i]} Locations')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f'Map for {partner_names[i]} with Circle')
                ax.legend()
                plt.show()"""

            count = 0
            j = 0
            for _, row in df_temp.iterrows():
                distance = self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon'])
                j += 1
                if distance <= radius:
                    count += 1
            count_inside.loc[i, 'Percentage'] = count / j
        if comparing == True:
            count_inside['Ranking'] = range(1, len(count_inside) +1)
        count_inside.index = count_inside['Partner']
        count_inside.drop(columns=['Partner'], inplace=True)
        count_inside = count_inside.sort_values(by=['Percentage'], ascending=False)
        count_inside.index.name = None
        return count_inside

    def features_bounding_box(self, df, matrix, comparing):
        matrix = self._clean_column_and_index_labels(matrix)
        company_name = matrix.index[0]
        partner_names = matrix.columns.unique().drop(company_name)
        company_df = df[df['name'] == company_name]
        partner_df = df[df['name'] != company_name]
        middlepoint = (company_df['lat'].mean(), company_df['lon'].mean())
        print(middlepoint)
        # Calculate distances from the midpoint to each delivery location
        distances = company_df.apply(
            lambda row: self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon']), axis=1)

        radius = distances.mean()

        mean_distance_within_circle = distances[distances <= radius].mean()
        area_of_circle = np.pi * (radius ** 2)
        delivery_density = len(distances[distances <= radius]) / area_of_circle
        mean_distance_to_midpoint = distances.mean()

        count_inside = pd.DataFrame({
            'Partner': partner_names,
            'Percentage': [0] * len(partner_names)
        })

        for i in range(len(partner_names)):
            df_temp = partner_df[partner_df['name'] == partner_names[i]]
            count = 0
            j = 0
            for _, row in df_temp.iterrows():
                distance = self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon'])
                j += 1
                if distance <= radius:
                    count += 1
            count_inside.loc[i, 'Percentage'] = count / j
        count_inside.index = count_inside['Partner']
        count_inside.drop(columns=['Partner'], inplace=True)
        count_inside = count_inside.sort_values(by=['Percentage'], ascending=False)
        count_inside.index.name = None
        # Return the computed features for further use
        return count_inside, mean_distance_within_circle, delivery_density, mean_distance_to_midpoint

    def k_means(self, df1, df, matrix, full_matrix, weighted=True):
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        minimal_clusters = len(df1[df1['name'] == chosen_company])
        partner_names = matrix1.columns.unique().drop(chosen_company)

        scaler = StandardScaler()
        df1[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df1[['lat', 'lon']])
        kmeans = KMeans(n_clusters=minimal_clusters, random_state=42)
        df1['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        sil_scores = {}
        b_s = {}
        a_s = {}
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i, random_state=42)
            df_new = df.copy()
            df_new['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])
            b_s[i] = self.average_distance_between_clusters(df_new, full_matrix)
            sil_scores[i], a_s[i] = self.mean_intra_cluster_distance(df_new, full_matrix, b_s[i], i)
        optimal_clusters = max(sil_scores, key=sil_scores.get) + minimal_clusters
        max_index = max(sil_scores, key=sil_scores.get) - minimal_clusters
        keys = list(a_s.keys())
        selected_key = keys[max_index]
        a = a_s[selected_key]
        b = b_s[selected_key]
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df1['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        clusters_company = df1[df1['name'] == chosen_company]['cluster'].unique()
        count_inside = pd.DataFrame({'Partner': partner_names, 'Percentage': [0] * len(partner_names)})

        percentage = []
        for partner in partner_names:
            df_temp = df1[df1['name'] == partner]
            df_temp2 = df_temp['cluster'].isin(clusters_company)
            if weighted== True:
                cluster_count = df1[df1['name'] == chosen_company]['cluster'].value_counts()
                overlap = df1[df1['name'] == partner]['cluster'].value_counts()
                weight = (overlap * cluster_count) / sum(cluster_count)
                real_weights = weight.fillna(0)
                weighted_cluster = df_temp2 * real_weights
                percentage.append((sum(weighted_cluster)) / len(df_temp2))
            else:
                percentage.append(sum(df_temp2) / len(df_temp))

        count_inside['Percentage'] = percentage
        ranking = count_inside.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        ranking.index = count_inside['Partner']
        ranking.drop(columns=['Partner'], inplace=True)
        ranking.index.name = None
        return ranking

    def features_k_means(self, df1, df, matrix, full_matrix, weighted=True):
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        minimal_clusters = len(df1[df1['name'] == chosen_company])
        partner_names = matrix1.columns.unique().drop(chosen_company)

        scaler = StandardScaler()
        df1[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df1[['lat', 'lon']])
        kmeans = KMeans(n_clusters=minimal_clusters, random_state=42)
        df1['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        sil_scores = {}
        b_s = {}
        a_s = {}
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i, random_state=42)
            df_new = df.copy()
            df_new['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])
            b_s[i] = self.average_distance_between_clusters(df_new, full_matrix)
            sil_scores[i], a_s[i] = self.mean_intra_cluster_distance(df_new, full_matrix, b_s[i], i)
        optimal_clusters = max(sil_scores, key=sil_scores.get) + minimal_clusters
        max_index = max(sil_scores, key=sil_scores.get) - minimal_clusters
        keys = list(a_s.keys())
        selected_key = keys[max_index]
        a = a_s[selected_key]
        b = b_s[selected_key]
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df1['cluster'] = kmeans.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        clusters_company = df1[df1['name'] == chosen_company]['cluster'].unique()
        count_inside = pd.DataFrame({'Partner': partner_names, 'Percentage': [0] * len(partner_names)})

        percentage = []
        for partner in partner_names:
            df_temp = df1[df1['name'] == partner]
            df_temp2 = df_temp['cluster'].isin(clusters_company)
            if weighted == True:
                cluster_count = df1[df1['name'] == chosen_company]['cluster'].value_counts()
                overlap = df1[df1['name'] == partner]['cluster'].value_counts()
                weight = (overlap * cluster_count) / sum(cluster_count)
                real_weights = weight.fillna(0)
                weighted_cluster = df_temp2 * real_weights
                percentage.append((sum(weighted_cluster)) / len(df_temp2))
            else:
                percentage.append(sum(df_temp2) / len(df_temp))

        count_inside['Percentage'] = percentage
        ranking = count_inside.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)

        # Compute average cluster size
        average_cluster_size = df1['cluster'].value_counts().mean()

        # Compute company centroid distance
        centroid_distances = []
        for cluster in df1['cluster'].unique():
            cluster_points = df1[df1['cluster'] == cluster][['lat_scaled', 'lon_scaled']]
            centroid = kmeans.cluster_centers_[cluster]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            centroid_distances.extend(distances)
        average_centroid_distance = np.mean(centroid_distances)

        # Compute average minimal and maximal distances within clusters
        min_distances = []
        max_distances = []
        for cluster in df1['cluster'].unique():
            cluster_points = df1[df1['cluster'] == cluster][['lat_scaled', 'lon_scaled']]

            # Only calculate distances if the cluster has more than one point
            if len(cluster_points) > 1:
                pairwise_dist = pdist(cluster_points)
                min_distances.append(np.min(pairwise_dist))
                max_distances.append(np.max(pairwise_dist))

        # Avoid computing mean of empty lists
        average_minimal_distance = np.mean(min_distances) if min_distances else 0
        average_maximal_distance = np.mean(max_distances) if max_distances else 0
        return ranking, average_cluster_size, average_centroid_distance, a, b, average_minimal_distance, average_maximal_distance

    def plot_clusters(self,df,k):
        plt.figure(figsize=(8, 6))

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

    def mean_intra_cluster_distance(self,df, distance_matrix,b,k):
        grouped = df.groupby('cluster')

        average_distances = {}
        sil_scores = np.zeros(k)
        for cluster, group in grouped:
            names = df.loc[group.index, 'name'].tolist()
            sub_matrix = distance_matrix.loc[names, names]
            mask = np.triu(np.ones(sub_matrix.shape, dtype=bool), k=1)
            upper_triangle_values = sub_matrix.values[mask]
            mean_distance = upper_triangle_values.mean() if len(names) > 1 else 0
            average_distances[cluster] = mean_distance
            sil_scores[cluster]= (b-mean_distance)/max(b,mean_distance)
        return np.mean(sil_scores)


    def average_distance_between_clusters(self,df, distance_matrix):
        grouped = df.groupby('cluster')
        cluster_distances = {}
        cluster_names = [cluster for cluster, _ in grouped]
        for i, cluster1 in enumerate(cluster_names):
            for j, cluster2 in enumerate(cluster_names):
                    names1 = df.loc[grouped.get_group(cluster1).index, 'name'].tolist()
                    names2 = df.loc[grouped.get_group(cluster2).index, 'name'].tolist()
                    sub_matrix = distance_matrix.loc[names1, names2]
                    dist_values = sub_matrix.values.flatten()
                    mean_distance_between_clusters = dist_values.mean()
                    cluster_distances[(cluster1, cluster2)] = mean_distance_between_clusters

        return np.mean(list(cluster_distances.values()))


    def dbscan(self, df, matrix):
        matrix = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix.index[0]
        scaler = StandardScaler()
        df[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df[['lat', 'lon']])

        dbscan = DBSCAN(eps=0.15, min_samples=2)
        df['cluster'] = dbscan.fit_predict(df[['lat_scaled', 'lon_scaled']])

        ranking = self.get_percentages(df, chosen_company)

        ranking.index = ranking['name']
        ranking.drop(columns=['name'], inplace=True)
        ranking.index.name = None
        return ranking

    def assign_noise_points(self, clusters, noise):
        centroids = clusters.groupby('cluster')[['lat_scaled', 'lon_scaled']].mean().values
        noise_coords = noise[['lat_scaled', 'lon_scaled']].values
        closest_clusters, _ = pairwise_distances_argmin_min(noise_coords, centroids)
        noise.loc[:, 'cluster'] = closest_clusters
        df_new = pd.concat([clusters, noise]).sort_index()
        return df_new

    def get_percentages(self, df_new, chosen_company):
        clusters = df_new[df_new['cluster'] != -1]
        noise = df_new[df_new['cluster'] == -1]

        if len(noise) != 0:
            df_new = self.assign_noise_points(clusters, noise)

        unique_clusters = df_new.loc[df_new['name'] == chosen_company, 'cluster'].unique()
        filtered_df = df_new[df_new['cluster'].isin(unique_clusters)]
        matching_counts = filtered_df.groupby('name').size()
        total_counts = df_new.groupby('name').size()
        percentages = (matching_counts / total_counts).reset_index()

        percentages.rename(columns={0: 'Percentage'}, inplace=True)
        percentages = percentages.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        percentages['Percentage'] = percentages['Percentage'].fillna(0)
        percentages = percentages[percentages['name'] != chosen_company]
        percentages["Rank"] = range(1, len(percentages) + 1)
        return percentages


    def compare(self,df,input_df1,j,input_df,method):
        correlations = []
        calculator = RoadDistanceCalculator()
        for i in range(len(df)):
            new_df = df.iloc[:, [i]]
            new_df.columns = ['Rankings']
            new_df = new_df[new_df['Rankings'] != len(df)]
            chosen_company = df.columns[i]
            distance_matrix = calculator.calculate_distance_matrix(
                input_df1, chosen_company=chosen_company,
                candidate_name=None, method="haversine", computed_distances_df=None)
            if method == "greedy":
                ranking = self.greedy(distance_matrix,comparing=True)
            elif method == "bounding box":
                ranking = self.bounding_box(input_df,distance_matrix,comparing=True)
            print('r',ranking)
            ranking_df1 = ranking['Ranking']
            ranking_df2 = new_df['Rankings']
            correlation, p_value = spearmanr(ranking_df1, ranking_df2)

            # Print the result
            print(f"Spearman's Rank Correlation: {correlation}")
            print(f"P-value: {p_value}")
            correlations.append(correlation)

        self.correlation_df[j] = correlations

    def features_dbscan(self, df1, df, matrix, full_matrix):
        eps = 0.6
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        scaler = StandardScaler()
        df[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df[['lat', 'lon']])
        df1[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df1[['lat', 'lon']])

        df_new = df1.copy()
        dbscan = DBSCAN(eps=eps, min_samples=2)
        df_new['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])
        percentages = self.get_percentages(df_new, chosen_company)

        clusters = df_new[df_new['cluster'] != -1]
        noise = df_new[df_new['cluster'] == -1]

        if len(noise) != 0:
            df_new = self.assign_noise_points(clusters, noise)

        cluster_midpoints = (
            df_new.groupby('cluster')[['lat_scaled', 'lon_scaled']].mean()
            .rename(columns={'lat_scaled': 'mid_lat', 'lon_scaled': 'mid_lon'})
        )

        df_new = df_new.merge(cluster_midpoints, how='left', left_on='cluster', right_index=True)

        dist_calc = RoadDistanceCalculator()

        df_new['distance_to_centroid'] = np.zeros(len(df_new))
        for i in range(len(df_new)):
            df_new['distance_to_centroid'][i] = dist_calc._calculate_distance_haversine((df_new['lat_scaled'][i], df_new['lon_scaled'][i]),
                                                           (df_new['mid_lat'][i], df_new['mid_lon'][i]))


        stats = df_new.groupby('name')[['distance_to_centroid']].agg(['mean', 'max', 'min'])
        stats.columns = ['avg_distance', 'max_distance', 'min_distance']
        stats.drop(chosen_company, inplace=True)
        percentages = percentages.merge(stats, how='left', left_on='name', right_index=True)
        percentages.rename(columns={'name': 'rank_dbscan', 'Percentage': 'percentage_dbscan'}, inplace=True)
        print(percentages)
        print(percentages.columns)
        return percentages

    def dbscan_tuning_true(self, df1, df, matrix):
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        scaler = StandardScaler()
        df1[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df1[['lat', 'lon']])

        dbscan = DBSCAN(eps=0.2, min_samples=2)
        df['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])
        df1['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        sp_scores = {}
        i = 0
        for e in eps:
            df_new = df1.copy()
            dbscan = DBSCAN(eps=e, min_samples=2)
            df_new['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])
            percentages = self.get_percentages(df_new, chosen_company)

            if percentages['Percentage'].eq(1).all():
                break
            sp_scores[i] = self.compare(percentages, chosen_company)
            i += 1

        argmax = np.argmax(list(sp_scores.values()))
        print("OPTIMAL EPS:", eps[argmax])
        new_df = df1.copy()
        dbscan = DBSCAN(eps=eps[argmax], min_samples=2)
        new_df['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])

        ranking = self.get_percentages(new_df, chosen_company)
        print(ranking)
        return ranking

    def dbscan_tuning_silscore(self, df1, df, matrix, full_matrix):
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        scaler = StandardScaler()
        df1[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df1[['lat', 'lon']])

        eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        sil_scores = {}
        i = 0
        for e in eps:
            df_new = df1.copy()
            dbscan = DBSCAN(eps=e, min_samples=2)
            df_new['cluster'] = dbscan.fit_predict(df1[['lat_scaled', 'lon_scaled']])
            if df_new['cluster'].eq(-1).all():
                continue
            percentages = self.get_percentages(df_new, chosen_company)

            if percentages['Percentage'].eq(1).all():
                break

            num_clusters = len(df_new['cluster'].unique())

            b = self.average_distance_between_clusters(df_new, full_matrix)
            sil_scores[i] = self.mean_intra_cluster_distance(df_new, full_matrix, b, num_clusters)
            i += 1
        argmax = np.argmax(list(sil_scores.values()))
        print("OPTIMAL EPS:", eps[argmax])


    def dbscan2(self, input_df_org, input_df, matrix, full_matrix):
        matrix1 = self._clean_column_and_index_labels(matrix)
        chosen_company = matrix1.index[0]
        print(chosen_company)

        my_list = []
        for i in range(len(full_matrix)):
            z=0
            my_list.append((full_matrix.iloc[i] < 50).sum() / len(full_matrix.iloc[i]))
            #print((full_matrix.iloc[i] < 8).sum() / len(full_matrix.iloc[i]))

        avg = sum(my_list) / len(my_list)
        #print("AVG: ", avg)
        total = avg * len(full_matrix)
        minsamples = round(total, 0)
        minsamples = int(minsamples)
        #print(minsamples)

        # Perform DBSCAN clustering
        eps = [3,4,5,6,7,8,9,10,11,12,13,14]  # Maximum distance for clustering
        min_samples = [2,3,4,5,6]   # Minimum number of points to form a cluster


        true_data = pd.read_excel("../Data/ranking_results_truck_cap_10_haversine_medium.xlsx", sheet_name=1, index_col=0)
        true_col_data = true_data[chosen_company]
        true_col_data.drop(chosen_company, inplace=True)
        true_col_data = true_col_data.sort_values()
        true = true_col_data.values

        highest_silscore = 0
        highest_corr = 0
        optimal_eps = 0
        optimal_ms =0
        for e in eps:
            for ms in min_samples:
                df_temp = input_df_org.copy()
                dbscan = DBSCAN(eps = e, min_samples = ms, metric = 'precomputed')
                clusters = dbscan.fit_predict(full_matrix)
                df_temp['cluster'] = clusters
                if df_temp['cluster'].eq(-1).all():
                    break

                df_temp = self.assign_noise_points2(df_temp, chosen_company, True)
                df_temp = df_temp[df_temp['cluster'] != -1]
                if (len(df_temp) / len(input_df_org)) < 0.5:
                    continue

                percentages = self.get_percentages(df_temp, chosen_company)
                percentages = percentages.set_index("name").loc[true_col_data.index].reset_index()

                test = percentages['Rank'].values
                correlation, p_value = spearmanr(true, test)

                #num_clusters = len(df_temp['cluster'].unique())

                #b = self.average_distance_between_clusters(df_temp, full_matrix)
                #sil_score = self.mean_intra_cluster_distance(df_temp, full_matrix, b, num_clusters)
                if correlation > highest_corr:
                    optimal_ms = ms
                    optimal_eps = e
                    highest_corr = correlation

                break
                """"if sil_score > highest_silscore:
                        optimal_ms = ms
                        optimal_eps = e
                        highest_silscore = sil_score"""

        print("OPTIMAL EPS:", optimal_eps, "OPTIMAL MS:", optimal_ms, 'HIGHEST SCORE:', highest_corr)
        df_temp = input_df_org.copy()
        dbscan = DBSCAN(eps = optimal_eps, min_samples = optimal_ms, metric = 'precomputed')
        clusters = dbscan.fit_predict(full_matrix)
        df_temp['cluster'] = clusters
        df_temp = self.assign_noise_points2(df_temp, chosen_company, True)
        df_temp = df_temp[df_temp['cluster'] != -1]
        percentages = self.get_percentages(df_temp, chosen_company)

        percentages.sort_values(by=['name'], inplace=True)
        test = percentages['Rank'].values

        print(percentages)

        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        clusters = dbscan.fit_predict(full_matrix)

        input_df_org['cluster'] = clusters
        input_df['cluster'] = clusters

        print(input_df_org['cluster'].value_counts())
        df_new = self.assign_noise_points2(input_df_org, chosen_company)

        df_new = df_new[df_new['cluster'] != -1]

        percentages = self.get_percentages(df_new, chosen_company)
        print(percentages)
        """


    def assign_noise_points2(self, df, chosen_company, company):
        if company is True:
            noise_df = df[(df['name'] == chosen_company) & (df['cluster'] == -1)].reset_index(drop=True)
        else:
            noise_df = df[df['cluster'] == -1].reset_index(drop=True)

        if noise_df.empty:
            return df

        clusters = df['cluster'].unique()
        clusters = sorted(clusters)
        centroids = []

        for cluster in clusters:
            if cluster != -1:  # Ignore noise
                cluster_points = df[df['cluster'] == cluster]
                centroid = (cluster_points['lat'].mean(), cluster_points['lon'].mean())
                centroids.append(centroid)

        centroids = np.array(centroids)

        for i in range(len(noise_df)):
            point = (noise_df['lat'][i], noise_df['lon'][i])
            distances = euclidean_distances([point], centroids)
            nearest_cluster = np.argmin(distances)
            noise_df.loc[i, 'cluster'] = clusters[nearest_cluster]

        for idx, row in noise_df.iterrows():
            # Match rows in df based on name, lat, and lon
            mask = (
                    (df['name'] == row['name']) &
                    (df['lat'] == row['lat']) &
                    (df['lon'] == row['lon'])
            )
            df.loc[mask, 'cluster'] = row['cluster']

        return df




#Third ranking algorithm
    def k_means(self, df_input, df_input_modified, full_dist_matrix, chosen_company):

        #Get correct format for matrix
        cleaned_matrix = self._clean_column_and_index_labels(full_dist_matrix)

        #Prep calculation
        minimal_clusters = len(df_input[df_input['name'] == chosen_company])
        partner_names = cleaned_matrix.columns.unique().drop(chosen_company)
        scaler = StandardScaler()
        df_input[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_input[['lat', 'lon']])

        #Determine number of clusters
        silhouette_scores_kmeans = {}
        b_s = {}
        a_s = {}
        for i in range(minimal_clusters, minimal_clusters + 6):
            kmeans = KMeans(n_clusters=i, random_state=42)
            df_new = df_input_modified.copy()
            df_new['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
            b_s[i] = self._average_distance_between_clusters(df_new, full_dist_matrix)
            silhouette_scores_kmeans[i], a_s[i] = self._mean_intra_cluster_distance(df_new, full_dist_matrix, b_s[i], i)

        #Determine optimal clusters
        optimal_clusters = max(silhouette_scores_kmeans, key=silhouette_scores_kmeans.get) + minimal_clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        df_input['cluster'] = kmeans.fit_predict(df_input[['lat_scaled', 'lon_scaled']])
        clusters_company = df_input[df_input['name'] == chosen_company]['cluster'].unique()

        #Creating ranking df
        ranking_df = pd.DataFrame({'Percentage Overlap': 0}, index=partner_names)

        #Get percentage overlap of clusters
        for partner in partner_names:
            partner_df = df_input[df_input['name'] == partner]
            partner_data = partner_df[partner_df['name'] == partner]
            is_in_cluster = partner_data['cluster'].isin(clusters_company)

            #Calculate and assign percentage overlap directly to ranking_df
            ranking_df.loc[partner, 'Percentage Overlap'] = is_in_cluster.mean()

        return ranking_df.sort_values(by=['Percentage Overlap'], ascending=False)

