#########################################################
#                                                       #
# Updated on: 24/01/2025                                #
# Updated by: Dennis                                    #
#                                                       #
#########################################################

from Input_Transformation.transforming_input import TransformInput
from Check_Candidate_Ranking_Performance.spearmans_methods import CalcSpearmans
import pandas as pd


if __name__ == "__main__":
    input_df = pd.read_csv("Data/medium.csv")

    check_road_proximity = True
    transformer = TransformInput(check_road_proximity=check_road_proximity)
    input_df_modified = transformer.execute_validations(input_df)
    accuracy = CalcSpearmans()
    company_names = input_df['name'].unique()
    accuracy.correlation_df = pd.DataFrame(index=company_names)
    correlation_df = pd.DataFrame(index=company_names)

    for i in [2,3,4,5,6,7,8,9,10,11,12]:
        method = "haversine"
        true_ranking = pd.read_excel(f"Solve_True_Rankings_VRP/Data_VRP_solved/{method}/ranking_results_truck_cap_{i}_haversine_medium.xlsx", sheet_name="Ranks")
        true_ranking.set_index('Unnamed: 0', inplace=True)
        accuracy.compute_spearman_greedy(true_ranking, input_df, input_df_modified,method ,i)
        #accuracy.compute_spearman_bounding_circle(true_ranking, input_df, input_df_modified,method ,i)
        #accuracy.compute_spearman_accuracy_k_means(true_ranking, input_df, input_df_modified,method ,i)

    print(accuracy.correlation_df)