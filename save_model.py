# save_model.py
from surprise import SVD, Dataset, Reader, dump
import pandas as pd

# Load your dataset
ratings = pd.read_csv('data/ratings_small.csv')

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()

# Train the model
model = SVD()
model.fit(trainset)

# Save the model
dump.dump('svd_model.pkl', algo=model)
print("✅ Model saved successfully.")
