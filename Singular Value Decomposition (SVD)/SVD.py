from MovieLens import MovieLens
from surprise import SVD

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

print('\nUser {} loved these movies...'.format(testSubject))
for ratings in lovedMovies:
    print(ml.getMovieName(ratings[0]))

print('\nUser {} hated these movies...'.format(testSubject))
for ratings in hatedMovies:
    print(ml.getMovieName(ratings[0]))

print('\nBuilding Recommendation model...')
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)
