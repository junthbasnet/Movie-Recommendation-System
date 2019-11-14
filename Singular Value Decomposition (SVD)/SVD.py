from MovieLens import MovieLens
from surprise import SVD


def buildAntiTestSetForUser(testSubject, trainSet):
    fill = trainSet.global_mean

    anti_testset = []

    u = trainSet.to_inner_uid(str(testSubject))
    user_items = set([j for (j, _) in trainSet.ur[u]])

    anti_testset += [(trainSet.to_raw_uid(u), trainSet.to_raw_iid(i), fill) for
                     i in trainSet.all_items() if
                     i not in user_items]
    return anti_testset


# Picking arbitrary test subject
testSubject = 85
ml = MovieLens()

print("Loading movie ratings...")
data = ml.loadMovieLensLatestSmall()

userRatings = ml.getUserRatings(testSubject)

lovedMovies = []
hatedMovies = []

for ratings in userRatings:
    if (float(ratings[1]) > 4.0):
        lovedMovies.append(ratings)
    if (float(ratings[1]) < 3.0):
        hatedMovies.append(ratings)

print('\nUser {} loved these movies:'.format(testSubject))
for ratings in lovedMovies:
    print(ml.getMovieName(ratings[0]))

print('\nUser {} hated these movies:'.format(testSubject))
for ratings in hatedMovies:
    print(ml.getMovieName(ratings[0]))

print('\nBuilding Recommendation model...')
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print('Computing recommendations...')
testSet = buildAntiTestSetForUser(testSubject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print('\nMovie Recommendation System recommends:')
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

for ratings in recommendations[:10]:
    print(ml.getMovieName(ratings[0]))
