from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)

    def getTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= actualRating):
                topN[int(userID)].append((movieID, estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN