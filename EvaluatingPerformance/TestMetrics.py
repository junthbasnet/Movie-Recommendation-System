from MovieLens import MovieLens
from RecommenderMetrics import RecommenderMetrics
from surprise import SVD
from surprise import KNNBaseline
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut

ml = MovieLens()

print('\nLoading movie ratings...')
data = ml.loadMovieLensLatestSmall()

print('\nBuilding Recommendation model...')
trainSet, testSet = train_test_split(data, test_size = .25, random_state = 1)

algo = SVD(random_state = 10)
algo.fit(trainSet)

print('\nComputing Recommendations...')
predictions = algo.test(testSet)

print('\nEvaluating accuracy of the recommendation model...')
print('RMSE (Root Mean Squared Error): {}'.format(RecommenderMetrics.RMSE(predictions)))
print('MAE (Mean Absolute Error): {}'.format(RecommenderMetrics.MAE(predictions)))