import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
LIKED_FILE = "liked_movies.txt"
def load_liked_movies():
    if os.path.exists(LIKED_FILE):
        with open(LIKED_FILE, "r") as f:
            return set(line.strip() for line in f.readlines())
    return set()
def save_liked_movies(liked_movies):
    with open(LIKED_FILE, "w") as f:
        for movie in liked_movies:
            f.write(movie + "\n")
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "liked_movies" not in st.session_state:
    st.session_state.liked_movies = load_liked_movies()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "details_open" not in st.session_state:
    st.session_state.details_open = set()
if "mood_mode" not in st.session_state:
    st.session_state.mood_mode = False
if "show_liked" not in st.session_state:
    st.session_state.show_liked = False
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#1c1c3c);
    background-size: 400% 400%;
}
.movie-card {
    background:#ffffff;
    color:#111;
    padding:16px;
    border-radius:14px;
    box-shadow:0 8px 20px rgba(0,0,0,0.25);
}
.details-card {
    background:#f8fafc;
    color:#111;
    padding:16px;
    border-radius:12px;
    margin-top:15px;
    box-shadow:0 4px 10px rgba(0,0,0,0.15);
    max-width: 100%;
    line-height: 1.6;
}
.details-card p {
    text-align: justify;
}
</style>
""", unsafe_allow_html=True)
OMDB_API_KEY = "89772b20"
DEFAULT_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"
@st.cache_data(ttl=86400)
def fetch_movie_details(title):
    try:
        url = "https://www.omdbapi.com/"
        params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("Response") == "False":
            raise ValueError
        poster = data.get("Poster")
        if poster == "N/A":
            poster = DEFAULT_POSTER
        return {
            "poster": poster,
            "plot": data.get("Plot", "N/A"),
            "rating": data.get("imdbRating", "N/A"),
            "director": data.get("Director", "N/A"),
            "year": data.get("Year", "N/A"),
            "runtime": data.get("Runtime", "N/A"),
            "actors": data.get("Actors", "N/A"),
            "genre": data.get("Genre", "N/A")
        }
    except:
        return {
            "poster": DEFAULT_POSTER,
            "plot": "N/A",
            "rating": "N/A",
            "director": "N/A",
            "year": "N/A",
            "runtime": "N/A",
            "actors": "N/A",
            "genre": "N/A"
        }
def has_valid_poster(title):
    details = fetch_movie_details(title)
    poster = details.get("poster", "")
    return poster and "placeholder.com" not in poster and poster != "N/A"
YOUTUBE_API_KEY = "AIzaSyBgAQYPwzm2gMNKYIvd3wZnOovGBDSjg6Q"
@st.cache_data(ttl=86400)
def fetch_trailer(movie_title):
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{movie_title} official trailer",
            "key": YOUTUBE_API_KEY,
            "maxResults": 1,
            "type": "video"
        }
        response = requests.get(url, params=params, timeout=5).json()
        if "items" in response and len(response["items"]) > 0:
            return response["items"][0]["id"]["videoId"]
        return None
    except:
        return None
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    movies = movies[['title','overview','genres','vote_average']].dropna()
    movies['genres'] = movies['genres'].apply(
        lambda x: ' '.join([g['name'] for g in ast.literal_eval(x)])
    )
    movies['content'] = movies['overview'] + " " + movies['genres']
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    all_genres = sorted(set(g for sub in movies['genres'].str.split() for g in sub))
    return movies, cosine_sim, all_genres
movies, cosine_sim, genre_list = load_data()
st.markdown("<h1 style='text-align:center;'>🎬 AI Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("---")
def personalized_recommendations(movies, cosine_sim, liked_movies, top_n=5):
    if not liked_movies:
        return []
    liked_indices = movies[movies['title'].isin(liked_movies)].index.tolist()
    if not liked_indices:
        return []
    avg_scores = cosine_sim[liked_indices].mean(axis=0)
    scores = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)
    recommendations = []
    for idx, _ in scores:
        title = movies.iloc[idx]['title']
        if title in liked_movies:
            continue
        if not has_valid_poster(title):
            continue
        recommendations.append(movies.iloc[idx])
        if len(recommendations) == top_n:
            break
    return recommendations
def get_personalized_recommendations(liked_movies):
    return personalized_recommendations(
        movies,
        cosine_sim,
        liked_movies
    )
if st.session_state.mood_mode:
    st.markdown("## 😊 Discover Movies by Your Emotions")
    mood = st.selectbox(
        "Select your mood",
        ["Happy", "Sad", "Romantic", "Excited", "Relaxed", "Thriller"]
    )
    min_rating = st.slider(
        "Minimum Rating",
        0.0, 10.0, 6.0,
        key="mood_rating"
    )
    mood_genre_map = {
        "Happy": ["Comedy", "Family", "Animation"],
        "Sad": ["Drama"],
        "Romantic": ["Romance"],
        "Excited": ["Action", "Adventure"],
        "Relaxed": ["Fantasy", "Music"],
        "Thriller": ["Thriller", "Crime", "Mystery"]
    }
    if st.button("🎭 Recommend by Mood"):
        st.session_state.active_mode = "mood"
        preferred_genres = mood_genre_map[mood]
        mood_recs = movies[
            movies["genres"].apply(
                lambda g: any(genre in g for genre in preferred_genres)
            )
        ]
        mood_recs = mood_recs[
            mood_recs["vote_average"] >= min_rating
        ].sort_values(by="vote_average", ascending=False).head(5)
        st.session_state.recommendations = [
            row for _, row in mood_recs.iterrows()
        ]
        st.session_state.show_results = True
if not st.session_state.mood_mode:
    c1, c2, c3 = st.columns(3)
    with c1:
        movie = st.selectbox("Select a movie", sorted(movies['title']))
    with c2:
        selected_genres = st.multiselect("Filter by Genre", genre_list)
    with c3:
        min_rating = st.slider("Minimum Rating",0.0, 10.0, 6.0,key="normal_rating")
    if st.button("✨ Recommend Movies"):
        st.session_state.active_mode = "movie"
        idx = movies[movies['title'] == movie].index[0]
        scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
        recs = []
        for i, _ in scores[1:]:
            m = movies.iloc[i]
            if m['vote_average'] < min_rating:
                continue
            if selected_genres and not any(g in m['genres'] for g in selected_genres):
                continue
            if not has_valid_poster(m['title']):
                continue
            recs.append(m)
            if len(recs) == 5:
                break
        st.session_state.recommendations = recs
        st.session_state.show_results = True
def toggle_details(title):
    if title in st.session_state.details_open:
        st.session_state.details_open.remove(title)
    else:
        st.session_state.details_open.add(title)
if st.session_state.show_results and st.session_state.active_mode:
    cols = st.columns(5)
    for col, m in zip(cols, st.session_state.recommendations):
        details = fetch_movie_details(m['title'])
        with col:
            st.image(details["poster"], use_container_width=True)
            st.markdown(f"""
            <div class="movie-card">
                <h4>{m['title']}</h4>
                <p>⭐ {m['vote_average']}</p>
                <p>{m['genres']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("👍 Like", key=f"like_{m['title']}"):
                st.session_state.liked_movies.add(m['title'])
                save_liked_movies(st.session_state.liked_movies)
            is_open = m['title'] in st.session_state.details_open
            st.button(
                "❌ Hide Details" if is_open else "📖 View Details",
                key=f"details_{m['title']}",
                on_click=toggle_details,
                args=(m['title'],)
            )
        if is_open:
            trailer_id = fetch_trailer(m['title'])
            st.markdown(f"""
            <div class="details-card">
                <h3>{m['title']}</h3>
                <b>🎬 Director:</b> {details['director']}<br>
                <b>📅 Year:</b> {details['year']}<br>
                <b>⏱ Duration:</b> {details['runtime']}<br>
                <b>🎭 Genre:</b> {details['genre']}<br>
                <b>🎥 Actors:</b> {details['actors']}<br><br>
                <b>📝 Plot:</b>
                <p>{details['plot']}</p>
            </div>
            """, unsafe_allow_html=True)
            if trailer_id:
                st.markdown("### 🎞 Official Trailer")
                st.video(f"https://www.youtube.com/watch?v={trailer_id}")
            else:
                st.info("Trailer not available 🎬")
if not st.session_state.mood_mode:
    if st.button("🎭 Discover Movies by Your Emotions"):
        st.session_state.mood_mode = True
        st.session_state.show_results = False
else:
    if st.button("⬅ Back to Recommend Movies"):
        st.session_state.mood_mode = False
        st.session_state.show_results = False
st.markdown("## 🤖 Personalized For You")
if st.session_state.liked_movies:
    if st.button("🎯 Get Personalized Movies"):
        st.session_state.active_mode = "personalized"
        st.session_state.details_open = set()
        st.session_state.recommendations = get_personalized_recommendations(
            st.session_state.liked_movies
        )
        st.session_state.show_results = True
else:
    st.info("Like some movies to unlock personalized recommendations ❤️")
st.markdown("## ❤️ View Liked Movies")
if st.session_state.liked_movies:
    selected_liked_movie = st.selectbox(
        "Your Liked Movies",
        sorted(st.session_state.liked_movies)
    )
    st.success(f"🎬 Selected Movie: {selected_liked_movie}")
    if st.button("🗑 Remove from Liked Movies"):
        st.session_state.liked_movies.remove(selected_liked_movie)
        save_liked_movies(st.session_state.liked_movies)
        st.success(f"Removed {selected_liked_movie} from liked movies")
else:
    st.info("No liked movies yet. Like some movies ❤️")
st.markdown("---")
st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)

