#different
import streamlit as st
import pandas as pd
import warnings
import folium
from streamlit_folium import st_folium
warnings.filterwarnings('ignore')


def createlookupdf(df):
    # Create a unique identifier for each row grouped by the 'name'
    df['name'] = df.groupby('name').cumcount().add(1).astype(str).radd(df['name']+"_")
    return df


class Dashboard:
    input_df = None
    vehicle_capacity = 0
    selected_company = ""
    heuristics_choice = ""

    execute_Ranking = False

    company_candidates = list()
    collaboration_company = ""


    def __init__(self):
        st.set_page_config(page_title = 'Collaboration Dashboard', page_icon = ":bar_chart", layout = 'wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

        # File uploader
        fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
        if fl is not None:
            if fl.name.endswith('.csv'):
                input_df = pd.read_csv(fl, encoding="ISO-8859-1")
            elif fl.name.endswith(('.xlsx', '.xls')):
                input_df = pd.read_excel(fl, engine='openpyxl')
            else:
                st.error("Unsupported file type!")
                st.stop()
            st.write(f"Uploaded file: {fl.name}")
            st.dataframe(input_df)

            st.sidebar.header("Choose your filter: ")

            #Company choice
            filters =list(input_df["name"].unique())
            selected_company = st.sidebar.selectbox("Pick your Company:", filters)

            #Vehicle capacity (max vehicle capacity is based on the biggest company (having 1 truck for the biggest company's delivery))
            max_locations = input_df["name"].value_counts().max()
            vehicle_range = range(1,max_locations+1)
            vehicle_capacity = st.sidebar.selectbox("Pick your Capacity:", list(vehicle_range))

            #Choice of Heuristic (Greedy)
            heuristics = list(["greedy", "boundingbox"])
            heuristics_choice = st.sidebar.selectbox("Pick your Heuristic:", heuristics)

            #Choice of Distance (haversine, osrm) FOR NOW ONLY CHOOSE HAVERSINE
            distance_choices = list(["haversine", "osrm (Docker Required)"])
            distance_choice = st.sidebar.selectbox("Pick your Distance:", distance_choices)

            if st.sidebar.button("Execute"):
                self.execute = True

    def execute(self):
        st.write("Executing Dashboard")

    def showMap(self, df, route):
        """
        Plots the locations and the shortest route on a map using Folium.

        :param df: DataFrame with columns ['name', 'lat', 'lon']
        :param route: List of names representing the shortest route
        """
        # Calculate the depot location (average of all latitudes and longitudes)
        depot_lat = df['lat'].mean()
        depot_lon = df['lon'].mean()
        depot_name = "Depot"

        # Add the depot to the DataFrame
        depot_df = pd.DataFrame({'name': [depot_name], 'lat': [depot_lat], 'lon': [depot_lon]})
        df = pd.concat([df, depot_df], ignore_index=True)

        # Initialize the map centered on the depot
        m = folium.Map(location=[depot_lat, depot_lon], zoom_start=7)

        # Define a color map for different company types
        company_colors = {}
        color_palette = ['red', 'blue', 'green', 'purple', 'orange']
        companies = df['name'].str.extract(r'^(.*?)(?:_\d+)?$')[0].unique()

        for i, company in enumerate(companies):
            company_colors[company] = color_palette[i % len(color_palette)]

        # Add markers for each location
        for _, row in df.iterrows():
            company = row['name'].split('_')[0] if '_' in row['name'] else row['name']
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=row['name'],
                icon=folium.Icon(color=company_colors.get(company, 'black'))
            ).add_to(m)

        # Add markers for the depot
        folium.Marker(
            location=[depot_lat, depot_lon],
            popup=depot_name,
            icon=folium.Icon(color='black', icon='home', prefix='fa')
        ).add_to(m)

        # Draw lines between the points in the route
        route_coords = []
        for loc_name in route + [depot_name]:  # Route ends back at the depot
            location = df[df['name'] == loc_name]
            if not location.empty:
                route_coords.append((location.iloc[0]['lat'], location.iloc[0]['lon']))

        folium.PolyLine(route_coords, color='black', weight=2.5, opacity=0.7).add_to(m)

        # Display the map using Streamlit
        st_folium(m, width=800, height=600)


# Create a Dashboard instance and show the map
dashboard = Dashboard()

if dashboard.execute_Ranking:
    heuristic = Ranking()
    if dashboard.heuristics_choice == "greedy":
        ranking = heuristic.greedy()
    if dashboard.heuristics_choice == "boundingbox":
        ranking = heuristic.boundingbox()

#dashboard.showMap(df, route)











