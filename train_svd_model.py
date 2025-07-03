# train_svd_model.py
import pandas as pd
from surprise import SVD, Dataset, Reader, dump

# Load data
ratings = pd.read_csv("data/ratings_small.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train and save model
model = SVD()
model.fit(trainset)
dump.dump("svd_model.pkl", algo=model)

print("✅ SVD model saved as svd_model.pkl")
