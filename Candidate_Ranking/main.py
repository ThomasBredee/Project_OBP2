from Candidate_Ranking.Rankings import CandidateRanking
from Algorithms.distance_calculator import RoadDistanceCalculator
import pandas as pd
import time


if __name__ == "__main__":
    input_df = pd.read_csv("../Data/manyLarge.csv")
    CHOSEN_COMPANY = "Dynamic Industries"
    #Get Distance matrix
    start_time = time.time()
    calculator = RoadDistanceCalculator()
    distance_matrix = calculator.calculate_distance_matrix(
        input_df, chosen_company=CHOSEN_COMPANY,
        candidate_name=None, method="haversine", computed_distances_df=None)
    algorithms = CandidateRanking()
    #algorithm1 = algorithms.greedy(distance_matrix)
    #algorithm2 = algorithms.bounding_box(input_df,distance_matrix)
    algorithm3 = algorithms.k_means(input_df,distance_matrix)