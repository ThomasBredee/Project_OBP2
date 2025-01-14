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
    df = None

    def __init__(self):
        st.set_page_config(page_title = 'Collaboration Dashboard', page_icon = ":bar_chart", layout = 'wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

        # File uploader
        fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
        if fl is not None:
            # Use the file-like object directly
            if fl.name.endswith('.csv'):
                df = pd.read_csv(fl, encoding="ISO-8859-1")
            elif fl.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(fl, engine='openpyxl')
            else:
                st.error("Unsupported file type!")
                st.stop()
            st.write(f"Uploaded file: {fl.name}")

            self.df = None

            selected_company = None

            st.sidebar.header("Choose your filter: ")
            filters = [None] + list(df["name"].unique())
            selected_company = st.sidebar.selectbox("Pick your Company:", filters)

            if not selected_company:
                df2 = df.copy()
            else:
                df2 = df[df["name"] == selected_company]

            if selected_company != None:
                st.dataframe(df2)

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

data = {
        "name": [
            "Pioneer Networks_1", "Pioneer Networks_2", "Pioneer Networks_3",
            "Pioneer Networks_4", "Pioneer Networks_5",
            "NextGen Technologies_1", "NextGen Technologies_2",
            "NextGen Technologies_3", "NextGen Technologies_4",
            "NextGen Technologies_5"
        ],
        "lat": [
            52.141899, 53.334869, 52.692184, 52.50573, 51.947071,
            52.897564, 52.322636, 52.074196, 53.173977, 52.1436
        ],
        "lon": [
            5.583035, 6.516187, 5.056780000000001, 4.98528, 6.02425,
            5.583602, 4.972053, 5.095043, 6.208481, 5.045227
        ]
    }
df = pd.DataFrame(data)

    # Example shortest route
route = [
        "Pioneer Networks_1", "Pioneer Networks_3", "NextGen Technologies_1",
        "NextGen Technologies_5", "Pioneer Networks_5", "Pioneer Networks_2"
    ]

    # Create a Dashboard instance and show the map
dashboard = Dashboard()
dashboard.showMap(df, route)











