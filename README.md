# Movie Recommendation Engine

This project is a Python-based movie recommendation system that provides two types of recommendations:
1.  **Personalized Recommendations:** Based on a user's past ratings (Collaborative Filtering).
2.  **Mood-Based Recommendations:** Based on movie genres.

The application is built with a user-friendly web interface using Streamlit.

## Features

*   **Collaborative Filtering:** Recommends movies to a user based on the ratings of other users with similar tastes.
*   **Genre-Based Filtering:** Recommends the top-rated movies within a selected genre.
*   **Interactive Web UI:** An easy-to-use interface built with Streamlit that allows users to get recommendations by entering their user ID or selecting a genre.

## How It Works

### Personalized Recommendations (Collaborative Filtering)

1.  **User-Item Matrix:** A matrix is created where rows represent users, columns represent movies, and the values are the ratings.
2.  **K-Nearest Neighbors (KNN):** The KNN algorithm is used to find a specified number of "neighbor" users who have the most similar rating patterns to the target user.
3.  **Recommendation Generation:** The system identifies movies that the neighbor users have rated highly but the target user has not yet seen. It then calculates the average rating for these movies among the neighbors and recommends the ones with the highest scores.

### Mood-Based Recommendations (Genre Filtering)

1.  **Genre Selection:** The user selects a genre from a dropdown list.
2.  **Movie Filtering:** The system filters the movie dataset to find all movies that belong to the selected genre.
3.  **Rating Aggregation:** It then calculates the average rating and the total number of ratings for each movie in that genre.
4.  **Top Recommendations:** To ensure quality, it filters out movies with a low number of ratings and then recommends the movies with the highest average rating.

## Project Structure

```
.
├── app.py                  # The main Streamlit web application
├── recommendation.py       # Contains the core recommendation logic
├── requirements.txt        # Lists the Python dependencies
└── data/
    ├── movies.csv          # The movie dataset
    └── ratings.csv         # The user ratings dataset
```

## How to Run the Application

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/writersrinivasan/MovieRecommendationEngine_Python.git
    cd MovieRecommendationEngine_Python
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application will open in your web browser, where you can interact with the recommendation system.
