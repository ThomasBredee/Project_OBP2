class AlgorithmEvaluation:
    def __init__(self):
        pass

    def evaluate_expected_gain(self, total_distance_single, total_distance_collab, num_orders_single):
        """
        Evaluate the expected gain in distance per order for collaboration.
        :param total_distance_single: Total distance for single company (no collaboration).
        :param total_distance_collab: Total distance for collaboration.
        :param num_orders_single: Total orders served by the single company.
        :return: Expected gain (distance saved per order).
        """
        if num_orders_single > 0:
            expected_gain = (total_distance_single - total_distance_collab) / num_orders_single
        else:
            expected_gain = 0

        print(f"Expected Gain (Distance Saved per Order): {expected_gain}")
        return expected_gain
