import streamlit as st
import pandas as pd
from recommendation import get_movie_recommendations, get_genre_recommendations

# Load the datasets
@st.cache_data
def load_data():
    ratings = pd.read_csv('data/ratings.csv')
    movies = pd.read_csv('data/movies.csv', engine='python', on_bad_lines='skip')
    return ratings, movies

ratings, movies = load_data()

st.title('Movie Recommendation System')

# --- Collaborative Filtering Section ---
st.header('Personalized Recommendations')
# User input
user_id = st.number_input('Enter your user ID', min_value=1, max_value=ratings['userId'].max(), value=1, step=1)

# Number of recommendations
num_recommendations = st.slider('Number of recommendations', 5, 20, 10)

if st.button('Get Personalized Recommendations'):
    recommendations, neighbor_users = get_movie_recommendations(user_id, ratings, movies, num_recommendations)
    st.subheader(f"Top {num_recommendations} recommendations for user {user_id}:")
    
    st.write(f"These recommendations are based on the viewing history of users: {', '.join(map(str, neighbor_users))}")

    if not recommendations.empty:
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}** ({row['genres']})")
    else:
        st.write("No recommendations found for this user.")

# --- Genre/Mood-Based Section ---
st.header('Mood-Based Recommendations')

# Get unique genres
all_genres = set()
for genre_list in movies['genres'].str.split('|'):
    all_genres.update(genre_list)
sorted_genres = sorted(list(all_genres))

# Genre selection
selected_genre = st.selectbox('Choose a genre based on your mood:', sorted_genres)

if st.button('Get Mood-Based Recommendations'):
    genre_recommendations = get_genre_recommendations(selected_genre, ratings, movies, num_recommendations)
    st.subheader(f"Top {num_recommendations} recommendations for the '{selected_genre}' genre:")

    if not genre_recommendations.empty:
        for index, row in genre_recommendations.iterrows():
            st.write(f"**{row['title']}** ({row['genres']})")
    else:
        st.write(f"No recommendations found for the '{selected_genre}' genre.")
