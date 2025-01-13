import requests

class Heuristics:

    def __init__(self,):
        self.session = requests.Session()

    def greedy(self, matrix):
        company_name = matrix.index[0]
        grouped_df = matrix.groupby(matrix.columns, axis=1).sum()
        grouped_df = grouped_df.drop(columns=company_name)
        ranking = grouped_df.sum(axis = 0)
        return ranking