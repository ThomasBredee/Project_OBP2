import streamlit as st
import pandas as pd
import warnings
import folium
from streamlit_folium import st_folium
warnings.filterwarnings('ignore')

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title='Collaboration Dashboard', page_icon=":bar_chart", layout='wide')
        st.title(" :bar_chart: Collaboration Dashboard")
        st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

        # Initialize session state variables
        if "execute_Ranking" not in st.session_state:
            st.session_state.execute_Ranking = False
        if "execute_VRP" not in st.session_state:
            st.session_state.execute_VRP = False
        if "show_vrp_button" not in st.session_state:
            st.session_state.show_vrp_button = False

        self.input_df = None
        self.vehicle_capacity = 0
        self.company_1 = None
        self.heuristics_choice = None
        self.company_2 = None

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
        st.dataframe(self.input_df)

        st.sidebar.header("Choose your filter: ")
        filters = list(self.input_df["name"].unique())
        self.company_1 = st.sidebar.selectbox("Pick your Company:", filters, key='company')

        max_locations = self.input_df["name"].value_counts().max()
        vehicle_range = range(1, max_locations + 1)
        self.vehicle_capacity = st.sidebar.selectbox("Pick your Capacity:", list(vehicle_range), key='vehicle_capacity')

        heuristics = list(["greedy", "boundingbox"])
        self.heuristics_choice = st.sidebar.selectbox("Pick your Heuristic:", heuristics, key='heuristics_choice')

        distance_choices = list(["haversine", "osrm (Docker Required)"])
        self.distance_choice = st.sidebar.selectbox("Pick your Distance:", distance_choices, key='distance_choice')

        if st.sidebar.button("Get Ranking"):
            st.session_state.execute_Ranking = True
            st.session_state.show_vrp_button = True

    def choose_candidate(self, df):
        st.write("Top 3 candidates:")
        candidates_top3 = df.index.unique()[:3]
        st.write(candidates_top3)

        self.company_2 = st.sidebar.selectbox("Pick your candidate:", df.index.unique(), key='candidates')

        if st.session_state.show_vrp_button:
            if st.sidebar.button("Get VRP"):
                st.session_state.execute_VRP = True

    def print_input(self, input):
        st.write(input)

    def Ranking(self, ranking):
        st.write(ranking)

    def Test(self):
        st.write("Test")

