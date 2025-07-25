# 🎬 MovieFind - A Movie Recommendation System

I made a collaborative-filtering movie recommendation system that suggests movies based on a user's input title using machine learning algorithms. This project was built as part of my university course in **Introduction to Artificial Intelligence**, and it was my first full-stack machine learning web app.

## 🔗 Live Demo

🌐 Try it here: [https://moviefind-recommender.onrender.com](https://moviefind-recommender.onrender.com)

---

## 📌 How It Works

1. **User Input**: The user types in a movie they like (e.g., *Twilight*).
2. **Similarity Search**: The system uses **cosine similarity** to find movies with similar patterns in user ratings.
3. **Machine Learning Models**:
   - **K-Nearest Neighbors (KNN)**: Finds similar movies based on rating vectors.
   - **Singular Value Decomposition (SVD)**: Enhances predictions using matrix factorization.
4. **Output**: The app displays a list of recommended movies along with their posters fetched from the **TMDB API**.

---

## 🛠️ Tech Stack

### 👩‍💻 Backend:
- **Python**
- **pandas**, **scikit-learn**, **Surprise** (for KNN and SVD)
- **Flask** (web framework)

### 🎨 Frontend:
- **HTML / CSS** (user interface)

### 📡 APIs:
- **TMDB API** (to fetch movie posters and details)

### 🚀 Deployment:
- **Render** (for hosting the web application)

---

## 📷 Screenshots

> *(You can insert images or GIFs here showing the app interface once ready)*

---

## 🧪 Model Evaluation

The model is evaluated using **Root Mean Squared Error (RMSE)** to measure how close the predicted ratings are to the actual ones.

---

## 💻 How to Run Locally

 1. Clone the Repository
 2. Install dependencies: pip install -r requirements.txt
 3. Add Your TMDB API Key: Create a .env file in the project root and add: TMDB_API_KEY=your_tmdb_api_key
 4. Run the App: python app.py
 5. Then click the link given to open your browser 

## 🙋‍♀️ About Me

Hi! I'm **Edna**, an Information Technology student with a growing interest in machine learning and artificial intelligence.  
This project challenged me in the best ways, from learning new libraries to deploying an app online.  

It was part of my school coursework, and I’m proud to have pushed through and brought it to life!

📫 Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/edna-omeni-172892276)

---

## 🤝 Contributions

Contributions, suggestions, and ideas are welcome!  
Feel free to fork the repo and submit a pull request. Let’s learn and build together 🚀
