import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

class Heuristics:

    def __init__(self,):
        self.session = requests.Session()

    def euclidean_distance(self,lat1, lon1, lat2, lon2):
        return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


    def greedy(self, matrix):
        company_name = matrix.index[0]
        grouped_df = matrix.groupby(matrix.columns, axis=1).sum()
        grouped_df = grouped_df.drop(columns=company_name)
        ranking = grouped_df.sum(axis = 0).sort_values(ascending = True).to_frame()
        print(type(ranking))
        ranking.columns = ['Total Distance']
        print(ranking)
        return ranking

    def bounding_box(self,df, matrix):
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

        #circle = plt.Circle((middlepoint[0], middlepoint[1]),  radius, color='r', fill=False)
        fig, ax = plt.subplots()

        circle = plt.Circle((middlepoint[0], middlepoint[1]), radius, color='r', fill=False, linestyle='--', label='Circle with radius')

        count_inside = pd.DataFrame({
            'Partner': partner_names,
            'Percentage': [0] * len(partner_names)  # All zeros in 'percentage' column
        })
        for i in range(len(partner_names)):
            df_temp = partner_df[partner_df['name'] == partner_names[i]]
            """"
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
            """
            count = 0
            j = 0
            for _, row in df_temp.iterrows():
                distance = self.euclidean_distance(middlepoint[0], middlepoint[1], row['lat'], row['lon'])
                j +=1
                if distance <= radius:
                    count += 1
            count_inside.loc[i, 'Percentage'] = count/j
        count_inside = count_inside.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        print(count_inside)




