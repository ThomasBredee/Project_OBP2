import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxRuntime
import matplotlib.pyplot as plt
from Algorithms.distance_calculator import RoadDistanceCalculator

class VRPSolver ():
    def addDepot(self, input_df, depot_lat, depot_lon):
        depot_row = {"name": "Depot", 'lat': depot_lat, 'lon': depot_lon}
        return pd.concat([pd.DataFrame([depot_row]), input_df], ignore_index=True)

    def getDistances(self, input_df, filter_comp1, filter_comp2):
        calculator = RoadDistanceCalculator()
        distance_matrix = calculator.calculate_distance_matrix(
            input_df, filter_comp1=filter_comp1, filter_comp2=filter_comp2, flavor="haversine"
        )
        return distance_matrix

    def buildModel(self, input_df, filter_comp1, filter_comp2, distance_matrix):
        COORDS = []
        current_names = []

        # input_df = self.addDepot(input_df, depot_lat, depot_lon)
        # distance_matrix = self.getDistances(input_df, filter_comp1, filter_comp2)
        # print(distance_matrix)
        for name, lat, lon in zip(input_df.name, input_df.lat, input_df.lon):
            if filter_comp1 in name or filter_comp2 in name or name == "Depot":
                COORDS.append((lat, lon))
                current_names.append(name)

        m = Model()
        m.add_vehicle_type(1, capacity=200)
        depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1], name=current_names[0])
        clients = [
            m.add_client(x=COORDS[idx][0], y=COORDS[idx][1], name=current_names[idx], delivery=1)
            for idx in range(1, len(COORDS))
        ]

        locations = [depot] + clients

        added_edges = set()
        for frm in locations:
            for to in locations:
                if frm.name != to.name:
                    edge = (frm.name, to.name)

                    # Check if the edge (or its reverse) has already been added
                    if edge not in added_edges and (to.name, frm.name) not in added_edges:
                        # Get the distance
                        distance = distance_matrix.loc[frm.name, to.name]
                        if isinstance(distance, pd.Series):
                            distance = distance.iloc[0]
                        # Add the edge to the set
                        added_edges.add(edge)

                        added_edges.add((to.name, frm.name))

                        # Print or add the edge
                        m.add_edge(frm, to, distance=distance)
                        m.add_edge(to, frm, distance=distance)
        return m, current_names

    def solve(self, m, max_runtime, display, current_names):
        res = m.solve(stop=MaxRuntime(max_runtime), display=display)  # one second

        solution = res.best
        route = [current_names[i] for i in solution.routes()[0].visits()]
        route.insert(0, current_names[0])  # Adding the depot at the start
        route.append(current_names[0])  # Adding the depot at the end
        print("Full route with depot:", route)

        return solution, route

    def plotRoute(self, route, input_df):
        route_df = input_df[input_df['name'].isin(route)]
        # Plot
        plt.figure(figsize=(10, 8))
        # Scatter plot for the locations
        plt.scatter(route_df['lon'], route_df['lat'], color='blue', label='Other Locations', alpha=0.6, marker='o')
        # Highlight the depot with a different marker
        depot = route_df[route_df['name'] == 'Depot']
        plt.scatter(depot['lon'], depot['lat'], color='red', label='Depot', alpha=1, marker='D', s=100)

        # Plot the optimal route
        for i in range(1, len(route)):
            start = route_df[route_df.name == route[i - 1]].iloc[0]
            end = route_df[route_df.name == route[i]].iloc[0]

            plt.arrow(start['lon'], start['lat'], end['lon'] - start['lon'], end['lat'] - start['lat'],
                      head_width=0.025, head_length=0.025, fc='green', ec='green')

        # Labels and Title
        plt.title('Optimal Route with Locations and Depot', fontsize=16)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Show legend
        plt.legend()

        # Show plot
        plt.grid(True)
        plt.show()

    def validateRoute(self, solution, route, distance_matrix, input_df):
        route_len = 0
        for i in range(1, len(route)):
            route_len += distance_matrix.loc[route[i - 1], route[i]]
        self.plotRoute(route, input_df)
        print("Calculated route length: ", route_len)
        print("Solution route length: ", solution.distance_cost())
        return route_len

