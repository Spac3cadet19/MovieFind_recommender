from flask import Flask, render_template, request
from recommender import get_recommendations, evaluate_model_rmse

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    recommended_movies = []
    message = ""
    rmse_score = evaluate_model_rmse()  # Evaluate once on load

    if request.method == "POST":
        movie = request.form.get("movie")
        if movie:
            recommended_movies, message, _ = get_recommendations(movie)

    return render_template("index.html",
                           recommended_movies=recommended_movies,  # ✅ full list of dicts
                           message=message,
                           rmse_score=rmse_score)

if __name__ == "__main__":
    app.run(debug=True)
