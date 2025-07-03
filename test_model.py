# test_model.py
from surprise import dump

# Load the saved model
_, model = dump.load("svd_model.pkl")

# Test prediction
prediction = model.predict(uid=1, iid=31)  # example: user 1, movie 31
print("Predicted rating:", prediction.est)
