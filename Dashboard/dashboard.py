#########################################################
#                                                       #
# Created on: 14/01/2025                                #
# Created by: Thomas                                    #
#                                                       #
#########################################################

import streamlit as st
import pandas as pd
import warnings
import folium
from streamlit_folium import st_folium
import time
warnings.filterwarnings('ignore')


class Dashboard:
    def __init__(self):
        #print("Initializing Dashboard")
        st.set_page_config(page_title='Collaboration Dashboard', page_icon=":bar_chart", layout='wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

        # Show and Executes True/False's
        if "execute_Ranking" not in st.session_state:
            st.session_state.execute_Ranking = False
        if "show_Ranking" not in st.session_state:
            st.session_state.show_Ranking = False
        if "execute_VRP" not in st.session_state:
            st.session_state.execute_VRP = False
        if "show_VRP" not in st.session_state:
            st.session_state.show_VRP = False
        if "update1" not in st.session_state:
            st.session_state.update1 = False
        if "firsttime1" not in st.session_state:
            st.session_state.firsttime1 = True
        if "update2" not in st.session_state:
            st.session_state.update2 = False
        if "firsttime2" not in st.session_state:
            st.session_state.firsttime2 = True

        # Dashboard input saves
        if "selected_candidate" not in st.session_state:
            st.session_state.selected_candidate = None
        if "vehicle_capacity" not in st.session_state:
            st.session_state.vehicle_capacity = None
        if "company_1" not in st.session_state:
            st.session_state.company_1 = None
        if "heuristic" not in st.session_state:
            st.session_state.heuristic = None
        if "distance" not in st.session_state:
            st.session_state.distance = None

        # Output saves
        if "ranking" not in st.session_state:
            st.session_state.ranking = None
        if "vrp_solver" not in st.session_state:
            st.session_state.vrp_solver = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "current_names" not in st.session_state:
            st.session_state.current_names = None
        if "solution" not in st.session_state:
            st.session_state.solution = None
        if "route" not in st.session_state:
            st.session_state.route = None
        if "expected_gain" not in st.session_state:
            st.session_state.expected_gain = None

        # Different versions of distance matrices
        if "reduced_distance_df" not in st.session_state:
            st.session_state.reduced_distance_df = None
        if "input_df" not in st.session_state:
            st.session_state.input_df = None
        if "input_df_numbered" not in st.session_state:
            st.session_state.input_df_numbered = None
        if "input_df_wdepot" not in st.session_state:
            st.session_state.input_df_wdepot = None
        if "full_matrix" not in st.session_state:
            st.session_state.full_matrix = None

        # File uploader
        fl = st.file_uploader(":file_folder: Upload a file", type=(["csv"]))

        #start_time = time.time()
        if fl is not None:
            self._process_file(fl)
        else:
            st.session_state.input_df = None

        #print("Processing file took {} seconds".format(time.time() - start_time))

    def _process_file(self, fl):
        if fl.name.endswith('.csv'):
            st.session_state.input_df = pd.read_csv(fl, encoding="ISO-8859-1")
            #print(st.session_state.input_df)
        elif fl.name.endswith(('.xlsx', '.xls')):
            st.session_state.input_df = pd.read_excel(fl, engine='openpyxl')
        else:
            st.error("Unsupported file type!")
            st.stop()

        st.write(f"Uploaded file: {fl.name}")

        st.sidebar.header("Choose your filter: ")
        filters = list(st.session_state.input_df["name"].unique())
        company = st.sidebar.selectbox("Pick your Company:", filters) #key='company')
        if company != st.session_state.company_1:
            st.session_state.company_1 = company
            st.session_state.update1 = True
            st.session_state.update2 = True

        # Use a number input for vehicle capacity with no maximum value
        st.session_state.vehicle_capacity = st.sidebar.number_input(
            "Pick your Capacity:",
            min_value=2,
                value=2,
            max_value=20,
            step=1,
            #key='vehicle_capacity'
        )

        heuristics = list(["greedy", "bounding_box", "k_means", "dbscan", "machine_learning"])
        st.session_state.heuristic = st.sidebar.selectbox("Pick your Heuristic:", heuristics)#, key='heuristics_choice')

        distance_choices = list(["haversine", "osrm"])
        st.session_state.distance = st.sidebar.selectbox("Pick your Distance:", distance_choices)#, key='distance_choice')

        if st.sidebar.button("Get Ranking"):
            st.session_state.update1 = False
            st.session_state.firsttime1 = True
            st.session_state.execute_Ranking = True
            st.session_state.show_Ranking = True
            st.session_state.display_VRP = False

    def display_ranking(self):

        st.write("### Top 3 Candidates")
        top3_candidates = st.session_state.ranking.index.unique()[:3]

        # Display candidates in the middle
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use the radio button to directly update the selected candidate in session state
            selected_candidate = st.radio("Select a Candidate:", top3_candidates)#, key="selected_candidate")
            if selected_candidate != st.session_state.selected_candidate:
                st.session_state.selected_candidate = selected_candidate
                st.session_state.update2 = True

            if st.session_state.update2 and st.session_state.firsttime2 == False:
                st.write('<div style="text-align: left; color: red; font-weight: bold; font-style: italic;">'
                            'Not up-to-date! <br>'
                            'Recalculate VRP. <br>'
                            '<br>'
                            '</div>',
                            unsafe_allow_html=True
                         )

        # Show the Solve VRP button if a candidate is selected
        if selected_candidate:
            if col2.button("Solve VRP"):
                #self.solve_vrp()
                st.session_state.execute_VRP = True
                st.session_state.update2 = False
                st.session_state.firsttime2 = False


    #def solve_vrp(self):
    #    st.session_state.execute_VRP = True

    def showmap(self, route_input, df_input):
        # Extract base company names (before the last underscore)
        def get_base_name(name):
            if "_" in name:
                return "_".join(name.split("_")[:-1])
            return name

        df_input['base_name'] = df_input['name'].apply(get_base_name)

        # Map each base company name to a unique color
        unique_companies = df_input['base_name'].unique()
        company_colors = ["blue", "green", "red"]
        company_color_map = {company: company_colors[i % len(company_colors)] for i, company in
                             enumerate(unique_companies)}

        # Assign unique colors for each truck's route
        truck_colors = [
            "blue", "orange", "red", "purple", "cyan",
            "magenta", "lime", "brown", "pink",
            "gold", "silver", "darkcyan", "indigo", "green",
            "black", "white", "lightblue", "darkgreen", "gray"
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

        if not depot_coords:
            st.error("Depot not found in the input DataFrame.")
            return

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

            # Draw a polyline for the truck's route if it has coordinates
            if route_coordinates:
                folium.PolyLine(
                    locations=route_coordinates,
                    color=route_colors[idx],
                    weight=2.5,
                    opacity=0.8,
                    tooltip=f"Truck {idx + 1}"
                ).add_to(route_map)
            #else:
             #   st.warning(f"Route #{idx + 1} has no valid locations and will not be displayed.")

        # Display the map in Streamlit
        st_folium(route_map, width=800, height=600)
