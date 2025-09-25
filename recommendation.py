import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def get_movie_recommendations(user_id, ratings, movies, num_recommendations=10):
    """
    Generates movie recommendations for a user based on collaborative filtering.

    Args:
        user_id (int): The ID of the user to generate recommendations for.
        ratings (pd.DataFrame): DataFrame with user ratings for movies.
        movies (pd.DataFrame): DataFrame with movie information.
        num_recommendations (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended movies.
    """
    # Create a user-item matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    user_movie_matrix_sparse = csr_matrix(user_movie_matrix.values)

    # Train a K-Nearest Neighbors model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_movie_matrix_sparse)

    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id].values.reshape(1, -1)

    # Find the nearest neighbors
    n_users = user_movie_matrix.shape[0]
    if num_recommendations >= n_users:
        num_recommendations = n_users - 1
        
    distances, indices = model_knn.kneighbors(user_ratings, n_neighbors=num_recommendations + 1)

    # Get the user indices of the nearest neighbors (excluding the user themselves)
    neighbor_user_indices = indices.squeeze()[1:]
    neighbor_user_ids = user_movie_matrix.index[neighbor_user_indices]

    # Get the movies rated by the neighbors
    neighbor_ratings = ratings[ratings['userId'].isin(neighbor_user_ids)]
    
    # Get movies the target user has already rated
    user_rated_movie_ids = ratings[ratings['userId'] == user_id]['movieId'].unique()

    # Recommend movies that neighbors liked but the user hasn't seen
    # We'll recommend movies with the highest average rating among neighbors
    recommended_movies_df = neighbor_ratings[~neighbor_ratings['movieId'].isin(user_rated_movie_ids)]
    
    # Calculate the average rating for each movie among the neighbors
    movie_recommendation_scores = recommended_movies_df.groupby('movieId')['rating'].mean()
    
    # Sort by score and get the top N
    top_movie_ids = movie_recommendation_scores.nlargest(num_recommendations).index
    
    # Get the movie titles
    recommended_movies = movies[movies['movieId'].isin(top_movie_ids)]

    return recommended_movies, neighbor_user_ids.tolist()

def get_genre_recommendations(genre, ratings, movies, num_recommendations=10):
    """
    Generates movie recommendations based on a selected genre.

    Args:
        genre (str): The genre to get recommendations for.
        ratings (pd.DataFrame): DataFrame with user ratings for movies.
        movies (pd.DataFrame): DataFrame with movie information.
        num_recommendations (int): The number of recommendations to return.

    Returns:
        pd.DataFrame: A DataFrame containing the recommended movies.
    """
    # Find movies in the selected genre
    genre_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    
    if genre_movies.empty:
        return pd.DataFrame()

    # Get the ratings for these movies
    genre_movie_ratings = ratings[ratings['movieId'].isin(genre_movies['movieId'])]

    # Calculate the average rating and number of ratings for each movie
    movie_stats = genre_movie_ratings.groupby('movieId').agg(
        mean_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()

    # Filter out movies with too few ratings to avoid niche, poorly-rated movies
    # We'll set a simple quantile-based threshold
    min_ratings_threshold = movie_stats['rating_count'].quantile(0.6)
    qualified_movies = movie_stats[movie_stats['rating_count'] >= min_ratings_threshold]

    # Sort by mean rating
    top_movies_ids = qualified_movies.sort_values(by='mean_rating', ascending=False).head(num_recommendations)['movieId']
    
    recommended_movies = movies[movies['movieId'].isin(top_movies_ids)]
    
    return recommended_movies


if __name__ == '__main__':
    # Load the datasets
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv', engine='python', on_bad_lines='skip')

    # Get recommendations for a user
    user_id = 1
    recommendations, neighbor_users = get_movie_recommendations(user_id, ratings, movies)
    print(f"Recommendations for user {user_id}:")
    print(recommendations[['title', 'genres']])
    print(f"\nBased on users: {neighbor_users}")

    # Get genre recommendations
    genre = 'Comedy'
    genre_recs = get_genre_recommendations(genre, ratings, movies)
    print(f"\nTop recommendations for {genre}:")
    print(genre_recs[['title', 'genres']])
