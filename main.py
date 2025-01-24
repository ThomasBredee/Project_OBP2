from Algorithms.distance_calculator import RoadDistanceCalculator
from Dashboard.dashboard import Dashboard
from Candidate_Ranking.ranking_methods import CandidateRanking
from Algorithms.solver_pyvrp import VRPSolver
from Expected_gain_prediction.prepare_input import PrepareInput
from Expected_gain_prediction.make_prediction import ModelPredictor
import streamlit as st
import time
import pandas as pd
import joblib

LONG_DEPOT = 5.26860985
LAT_DEPOT = 52.2517788

FIRST_TIME = True

if __name__ == "__main__":

    start_time_overall = time.time()

    dashboard = Dashboard()
    algorithm = RoadDistanceCalculator()
    heuristic = CandidateRanking()

    if st.session_state.update1 and st.session_state.firsttime1 == False:
        st.sidebar.write(
            '<div style="text-align: center; color: red; font-weight: bold; font-style: italic;">'
            'Not up-to-date! <br>'
            'Recalculate Ranking and VRP.'
            '</div>',
            unsafe_allow_html=True
        )

    # Check if ranking needs to be executed
    if st.session_state.execute_Ranking and st.session_state.input_df is not None:
        st.session_state.firsttime1 = False
        calc_input_df = st.session_state.input_df.copy()
        st.session_state.input_df_numbered = st.session_state.input_df.copy()
        st.session_state.input_df_numbered['name'] = calc_input_df.groupby('name').cumcount().add(1).astype(str).radd(calc_input_df['name'] + "_")

        # Calculate the distance matrix
        start_time = time.time()
        st.session_state.reduced_distance_df = algorithm.calculate_distance_matrix(
            st.session_state.input_df_numbered, st.session_state.company_1, method="haversine"
        )
        print("Distance matrix Ranking took:", round(time.time() - start_time, 4), "seconds")

        start_time = time.time()
        # Generate ranking based on heuristic
        if st.session_state.heuristic == "greedy":
            st.session_state.ranking = heuristic.greedy(st.session_state.reduced_distance_df, comparing = False)
        elif st.session_state.heuristic == "bounding_box":
            st.session_state.ranking = heuristic.bounding_box(st.session_state.input_df,
                                                              st.session_state.reduced_distance_df, comparing = False)
        elif st.session_state.heuristic == "k_means":
            st.session_state.full_matrix = algorithm.calculate_square_matrix(st.session_state.input_df_numbered)
            st.session_state.ranking = heuristic.k_means(st.session_state.input_df,
                                                         st.session_state.input_df_numbered,
                                                         st.session_state.reduced_distance_df,
                                                         st.session_state.full_matrix, weighted = False)
        elif st.session_state.heuristic == "dbscan":
            st.session_state.full_matrix = algorithm.calculate_square_matrix(st.session_state.input_df)
            st.session_state.ranking = heuristic.dbscan(st.session_state.input_df,
                                                         st.session_state.reduced_distance_df)
        elif st.session_state.heuristic == "machine_learning":
            df_input_depot = algorithm.add_depot(st.session_state.input_df_numbered, LAT_DEPOT,
                                                 LONG_DEPOT)
            ranking = CandidateRanking()
            greedy_ranking = ranking.greedy(st.session_state.reduced_distance_df, comparing=False)

            ###make prediction df
            prep = PrepareInput()
            prediction_df = prep.prep_greedy(df_input_depot, st.session_state.company_1, greedy_ranking.index,
                                             "haversine",
                                             st.session_state.reduced_distance_df, st.session_state.vehicle_capacity,
                                             greedy_ranking)
            path = "Expected_gain_models/osrm/TrainedModels/RF/"
            # Load the saved scaler from a file
            scaler = joblib.load(f"{path}scaler_greedy_osrm.pkl")
            # Load the saved model
            model = joblib.load(f"{path}random_forest_model_greedy_osrm.pkl")
            predictor = ModelPredictor(model, scaler)

            predicted_df = predictor.predict_for_candidates(prediction_df)
            st.session_state.ranking = predicted_df.sort_values(by='Prediction')


        #print(st.session_state.ranking)
        st.session_state.execute_Ranking = False
        st.session_state.show_Ranking = True
        print("Ranking took:",  round(time.time() - start_time,4), "seconds")

    if st.session_state.show_Ranking and st.session_state.input_df is not None:
        # Display the ranking
        dashboard.display_ranking()
        csv_file, file_name = dashboard.download(type="ranking")
        st.sidebar.download_button(
            label='Download Ranking',
            data=csv_file,
            file_name=file_name,
            mime="text/csv"
        )

    # Check if VRP execution is triggered
    if st.session_state.execute_VRP and st.session_state.selected_candidate and st.session_state.input_df is not None:
        start_time = time.time()
        st.session_state.input_df_wdepot = algorithm.add_depot(st.session_state.input_df_numbered, LAT_DEPOT, LONG_DEPOT)

        distance_matrix_vrp = algorithm.calculate_distance_matrix(
            st.session_state.input_df_wdepot,
            chosen_company=st.session_state.company_1,
            candidate_name=st.session_state.selected_candidate,
            method=st.session_state.distance,
            computed_distances_df=st.session_state.reduced_distance_df,
        )
        print("Distance matrix VRP took:", round(time.time() - start_time,4), "seconds")
        start_time = time.time()

        # Solve the VRP
        st.session_state.vrp_solver = VRPSolver()
        st.session_state.model, st.session_state.current_names = st.session_state.vrp_solver.build_model(
            st.session_state.input_df_wdepot,
            st.session_state.company_1,
            st.session_state.selected_candidate,
            distance_matrix_vrp,
            st.session_state.vehicle_capacity,
        )

        st.session_state.solution, st.session_state.route = st.session_state.vrp_solver.solve(st.session_state.model, max_runtime=1, display=False,
                                                            current_names=st.session_state.current_names)

        st.session_state.execute_VRP = False
        st.session_state.show_VRP = True

        print("VRPSolver took:  ", round(time.time() - start_time,4), "seconds")

    if st.session_state.show_VRP and st.session_state.input_df is not None:



        st.write("### VRP Solution")
        st.session_state.solution_print = pd.DataFrame(st.session_state.route, index=[f"Route_{i}" for i in range(len(st.session_state.route))])

        col1, col2, col3 = st.columns([15, 1, 2.5])
        with col1:
            st.write(st.session_state.solution_print)
        with col3:
            csv_file, file_name = dashboard.download(type="vrp")
            st.download_button(
                label='Download VRP',
                data=csv_file,
                file_name=file_name,
                mime="text/csv"
            )

        #start_time_test = time.time()
        dashboard.showmap(st.session_state.route, st.session_state.input_df_wdepot)
        #print("\nTESTING UNITS", time.time() - start_time_test, "seconds\n")

    print("Overall re-initialization time took:", round(time.time() - start_time_overall,4), "seconds")