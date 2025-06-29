import pandas as pd
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import requests

# Optional: For model evaluation
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

TMDB_API_KEY = '576b2e4b9d41e15b43b1602bd4e22f64'  # Replace with your actual key

def get_poster_url(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0].get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return None

# Load data
ratings = pd.read_csv("data/ratings_small.csv")
movies = pd.read_csv("data/movies_metadata.csv", low_memory=False)
movies = movies[['id', 'title']].dropna()
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies = movies.dropna()
movies['id'] = movies['id'].astype(int)

ratings = ratings.merge(movies, left_on='movieId', right_on='id')

# Filter for performance
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

# Train KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_movie_matrix.T)

# Main function
def get_recommendations(movie_title, n_recommendations=10):
    movie_list = user_movie_matrix.columns.tolist()
    matched = process.extractOne(movie_title, movie_list)

    if not matched:
        return [], f"No match found for '{movie_title}'", None

    matched_title = matched[0]
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

    return recommendations, f"Showing recommendations for: {matched_title}", None

# Optional RMSE evaluator
def evaluate_model_rmse():
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    return round(rmse, 4)
