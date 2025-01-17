import streamlit as st
import pandas as pd
import warnings
import folium
from streamlit_folium import st_folium
warnings.filterwarnings('ignore')


class Dashboard:
    def __init__(self):
        print("Initializing Dashboard")
        st.set_page_config(page_title='Collaboration Dashboard', page_icon=":bar_chart", layout='wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

        # Initialize session state variables
        if "execute_Ranking" not in st.session_state:
            st.session_state.execute_Ranking = False
        if "execute_VRP" not in st.session_state:
            st.session_state.execute_VRP = False
        if "selected_candidate" not in st.session_state:
            st.session_state.selected_candidate = None

        self.input_df = None
        self.vehicle_capacity = 0
        self.company_1 = None
        self.heuristics_choice = None

        # File uploader
        fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
        if fl is not None:
            self.process_file(fl)

    def process_file(self, fl):
        if fl.name.endswith('.csv'):
            self.input_df = pd.read_csv(fl, encoding="ISO-8859-1")
        elif fl.name.endswith(('.xlsx', '.xls')):
            self.input_df = pd.read_excel(fl, engine='openpyxl')
        else:
            st.error("Unsupported file type!")
            st.stop()

        st.write(f"Uploaded file: {fl.name}")

        st.sidebar.header("Choose your filter: ")
        filters = list(self.input_df["name"].unique())
        self.company_1 = st.sidebar.selectbox("Pick your Company:", filters, key='company')

        # Use a number input for vehicle capacity with no maximum value
        self.vehicle_capacity = st.sidebar.number_input(
            "Pick your Capacity:",
            min_value=2,
            value=2,
            step=1,
            key='vehicle_capacity'
        )

        heuristics = list(["greedy", "boundingbox"])
        self.heuristics_choice = st.sidebar.selectbox("Pick your Heuristic:", heuristics, key='heuristics_choice')

        distance_choices = list(["haversine", "osrm (Docker Required)"])
        self.distance_choice = st.sidebar.selectbox("Pick your Distance:", distance_choices, key='distance_choice')

        if st.sidebar.button("Get Ranking"):
            st.session_state.execute_Ranking = True

    def display_ranking(self, ranking_df):
        st.write("### Top 3 Candidates")
        top3_candidates = ranking_df.index.unique()[:3]

        # Display candidates in the middle
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use the radio button to directly update the selected candidate in session state
            selected_candidate = st.radio("Select a Candidate:", top3_candidates, key="selected_candidate")

        # Show the Solve VRP button if a candidate is selected
        if selected_candidate:
            if col2.button("Solve VRP"):
                self.solve_vrp()

    def solve_vrp(self):
        st.session_state.execute_VRP = True

    def showmap(self, route_input, df_input):
        # Extract base company names (before the last underscore)
        def get_base_name(name):
            if "_" in name:
                return "_".join(name.split("_")[:-1])
            return name

        df_input['base_name'] = df_input['name'].apply(get_base_name)

        # Map each base company name to a unique color
        unique_companies = df_input['base_name'].unique()
        company_colors = ["blue", "green", "red", ]
        company_color_map = {company: company_colors[i % len(company_colors)] for i, company in
                             enumerate(unique_companies)}

        # Assign unique colors for each truck's route
        truck_colors = [
            "darkblue", "darkgreen", "darkorange", "lightred", "darkpurple", "black",
            "blue", "purple", "orange", "darkred",
            "cyan", "magenta", "lime", "brown", "pink", "teal",
            "gold", "silver", "darkcyan", "indigo"
        ]
        route_colors = [truck_colors[i % len(truck_colors)] for i in range(len(route_input))]

        # Create a folium map centered on the depot of the first route
        depot_coords = None
        for truck_route in route_input:
            first_location = truck_route[0]
            depot_row = df_input[df_input['name'] == first_location]
            if not depot_row.empty:
                depot_coords = (depot_row.iloc[0]['lat'], depot_row.iloc[0]['lon'])
                break

        # if not depot_coords:
        #    st.error("Depot not found in the input DataFrame.")
        #    return

        route_map = folium.Map(location=depot_coords, zoom_start=8)

        # Process each truck's route
        for idx, truck_route in enumerate(route_input):
            route_coordinates = []
            for location in truck_route:
                location_row = df_input[df_input['name'] == location]
                if location_row.empty:
                    st.warning(f"Location '{location}' not found in the input DataFrame.")
                    continue

                lat, lon = location_row.iloc[0]['lat'], location_row.iloc[0]['lon']
                route_coordinates.append((lat, lon))

                # Add markers for each location in the truck's route
                base_name = get_base_name(location)
                color = company_color_map.get(base_name, "gray")
                folium.Marker(
                    location=(lat, lon),
                    popup=f"{location} ({base_name})",
                    icon=folium.Icon(color=color if location != "Depot" else "black"),
                ).add_to(route_map)

            # Draw a polyline for the truck's route
            folium.PolyLine(
                locations=route_coordinates,
                color=route_colors[idx],
                weight=2.5,
                opacity=0.8,
                tooltip=f"Truck {idx + 1}"
            ).add_to(route_map)

        # Display the map in Streamlit
        st_folium(route_map, width=800,height = 600)


