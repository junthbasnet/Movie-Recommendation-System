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
print('\n 1. RMSE (Root Mean Squared Error): {}'.format(RecommenderMetrics.RMSE(predictions)))
print('2. MAE (Mean Absolute Error): {}'.format(RecommenderMetrics.MAE(predictions)))

print('\nEvaluating top 10 recommendations...')

LOOCV = LeaveOneOut(n_splits=1, random_state=1)

for trainSet, testSet in LOOCV.split(data):
    print('\nComputing Recommendation with LOOCV...')

    algo.fit(trainSet)
    leftOutPredictions = algo.test(testSet)

    bigTestSet = trainSet.build_anti_testset()
    allPredictions = algo.test(bigTestSet)

    print('\nComputing 10 recommendations per user...')
    topNPredicted = RecommenderMetrics.getTopN(allPredictions, n=10)

    print('\n Hit Rate: {}'.format(RecommenderMetrics.hitRate(topNPredicted, leftOutPredictions)))


