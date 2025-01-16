import pandas as pd

class CandidateRanking:
    def __init__(self):
        pass

    # Define the function to remove everything after '_' in column names and index
    @staticmethod
    def _clean_column_and_index_labels(matrix):
        matrix_copy = matrix.copy()  # Create a copy to avoid modifying the original
        matrix_copy.columns = [col.split('_')[0] for col in matrix_copy.columns]
        matrix_copy.index = [idx.split('_')[0] for idx in matrix_copy.index]
        return matrix_copy

    def greedy(self, matrix):
        # Clean column names and index
        matrix_new = self._clean_column_and_index_labels(matrix)

        company_name = matrix_new.index[0]
        grouped_df = matrix_new.groupby(matrix_new.columns, axis=1).sum()
        grouped_df = grouped_df.drop(columns=company_name)
        ranking = grouped_df.sum(axis=0).sort_values(ascending=True).to_frame()
        ranking.columns = ['Total Distance']
        ranking = round(ranking, 0)
        return ranking
