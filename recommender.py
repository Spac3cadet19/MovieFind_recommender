import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import requests

# Optional: For model evaluation
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

from surprise import dump

# Load saved SVD model
try:
    _, svd_model = dump.load("svd_model.pkl")
except:
    svd_model = None  # fallback if model isn't found


# TMDB API Key
TMDB_API_KEY = '576b2e4b9d41e15b43b1602bd4e22f64'

# Get poster from TMDB
def get_poster_url(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    
    try:
        response = requests.get(url, timeout=5)  # wait max 5 seconds
        response.raise_for_status()

        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    
    except requests.exceptions.Timeout:
        print(f"⚠️ TMDB API timed out for: {movie_title}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error fetching poster for '{movie_title}': {e}")
    
    # Fallback image to prevent app crash
    return "https://via.placeholder.com/200x300?text=No+Poster"

# Don't forget to keep this line too
from surprise import dump


# Load the pre-trained model


# Load and clean data
ratings = pd.read_csv("data/ratings_small.csv")
movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
movies = movies[['id', 'title']].dropna()
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna()
movies['id'] = movies['id'].astype(int)

ratings = ratings.merge(movies, left_on='movieId', right_on='id')

# Filter top movies/users for performance
top_movies = ratings['title'].value_counts().nlargest(1000).index
filtered_ratings = ratings[ratings['title'].isin(top_movies)]
top_users = filtered_ratings['userId'].value_counts().nlargest(10000).index
filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(top_users)]

# Create user-movie matrix
user_movie_matrix = filtered_ratings.pivot_table(
    index='userId',
    columns='title',
    values='rating'
).fillna(0)

# Train KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.T)

# Get recommendations
def get_recommendations(movie_title, n_recommendations=10):
    movie_list = user_movie_matrix.columns.tolist()
    matched = process.extractOne(movie_title, movie_list)

    if not matched or matched[1] < 60:
        return [], f"❌ No close match found for '{movie_title}'. Try another title.", None

    matched_title = matched[0]
    similarity_score = matched[1]
    movie_index = movie_list.index(matched_title)

    distances, indices = model_knn.kneighbors(
        [user_movie_matrix.T.iloc[movie_index]],
        n_neighbors=n_recommendations + 1
    )

    recommendations = []
    for i in range(1, len(distances[0])):
        recommended_title = user_movie_matrix.T.index[indices[0][i]]
        score = round(1 - distances[0][i], 2)
        poster = get_poster_url(recommended_title)
        recommendations.append({
            'title': recommended_title,
            'score': score,
            'poster': poster
        })

    if movie_title.lower() == matched_title.lower():
        message = f"Showing recommendations for: {matched_title}"
    else:
        message = f"🔍 Closest match found for '{movie_title}' is '{matched_title}'. Showing recommendations..."

    return recommendations, message, None

# Evaluate with SVD
def evaluate_model_rmse():
    if svd_model is None:
        return "⚠️ No pre-trained SVD model found."

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    predictions = svd_model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return round(rmse, 4)

