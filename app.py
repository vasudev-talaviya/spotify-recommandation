import streamlit as st
import pickle as pk
import pandas as pd
import numpy as np
import requests
import base64
import re 

# We need to import the Google API client library
# from googleapiclient.discovery import build

# --- Custom CSS for a more polished look ---
st.markdown(
    """
    <style>
    :root {
        --spotify-green: #1DB954;
        --dark-bg: #121212;
        --card-bg: #181818;
        --text-color: #E0E0E0;
        --muted-text: #A0A0A0;
    }
    body {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, var(--dark-bg), #000000);
        color: var(--text-color);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 950px;
    }
    h1 {
        color: var(--spotify-green);
        text-align: center;
        font-weight: 800;
        letter-spacing: -1px;
    }
    h3 {
        color: var(--text-color);
        font-weight: 600;
        border-bottom: 2px solid var(--spotify-green);
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .st-emotion-cache-18ni7ap {
        color: var(--muted-text);
        font-weight: 600;
    }
    .st-emotion-cache-12m1768 {
        border-radius: 12px;
        border: 1px solid #303030;
    }
    .card-container {
        background-color: var(--card-bg);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        display: flex;
        flex-direction: column;
        height: 100%;
        border: 1px solid #282828;
    }
    .card-container:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    }
    .card-img {
        width: 100%;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .card-body {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .card-title {
        color: var(--spotify-green);
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 0.25rem;
    }
    .card-text {
        font-size: 0.9rem;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .spotify-link {
        color: white;
        background-color: var(--spotify-green);
        border-radius: 20px;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        font-weight: bold;
        display: block;
        margin-top: auto;
        transition: background-color 0.2s;
    }
    .spotify-link:hover {
        background-color: #1ed760;
    }
    .st-emotion-cache-k3q0h6 {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. Load Resources and Authentication ---
@st.cache_resource
def get_spotify_access_token():
    """Fetches a Spotify API access token using credentials from secrets.toml."""
    try:
        client_id = st.secrets["SPOTIPY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
    except KeyError as e:
        st.error(f"Error: Missing Spotify credential '{e}' in .streamlit/secrets.toml. Please configure your secrets file.")
        st.stop()

    try:
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
        
        auth_response = requests.post(auth_url, {
            'grant_type': 'client_credentials',
        }, headers={"Authorization": f"Basic {auth_header}"})

        auth_response.raise_for_status()
        return auth_response.json().get('access_token')

    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
        st.error(f"Error authenticating with Spotify API: {e}. Please check your SPOTIPY_CLIENT_SECRET and SPOTIPY_CLIENT_ID in .streamlit/secrets.toml.")
        return None

@st.cache_resource
def load_model_and_data():
    """Loads the pre-trained machine learning model and data."""
    try:
        with open("./models/spotify_group_model.pkl", "rb") as f:
            model = pk.load(f)
        data = pk.load(open("./models/spotify_data.pkl", 'rb'))
        return model, data
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found: {e}")
        return None, None

# Load the resources at the start
model, data = load_model_and_data()
access_token = get_spotify_access_token()


# --- 2. Define the Recommendation Logic ---
@st.cache_data
def search_for_song_details(songs_info, access_token):
    """Searches for song details from the Spotify API using track name and artist."""
    recommended_tracks = []
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    for song_info in songs_info:
        track_name = song_info['name']
        artist_name_raw = song_info.get('artists', 'unknown')
        
        if isinstance(artist_name_raw, str):
            artist_name = re.sub(r"[\[\]'']", '', artist_name_raw)
        else:
            artist_name = 'unknown'

        query = f"track:{track_name} artist:{artist_name}"
        params = {
            'q': query,
            'type': 'track',
            'limit': 1
        }
        
        try:
            api_url = 'https://api.spotify.com/v1/search'
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            
            results = response.json().get('tracks', {}).get('items', [])
            
            if results:
                recommended_tracks.append(results[0])
            else:
                st.warning(f"Could not find exact details for: '{track_name}' by '{artist_name}'.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error searching for song details from Spotify API: {e}")
            
    return recommended_tracks

@st.cache_data
def recommend_songs(input_features, num_recommendations=10):
    """Recommends songs from the predicted cluster."""
    if model is None or data is None:
        st.error("Model or data failed to load. Cannot recommend songs.")
        return None

    predicted_cluster = model.predict(input_features)[0]
    
    recommended_df = data[data['cluster'] == predicted_cluster].sample(n=num_recommendations, random_state=42)
    
    songs_info = recommended_df[['name', 'artists']].to_dict('records')
    
    recommended_tracks = search_for_song_details(songs_info, access_token)
    
    return recommended_tracks


# --- 3. Main App UI Layout ---
st.title("🎵 Spotify Music Recommender")

st.markdown("""
<div style="text-align: center; color: var(--muted-text); font-style: italic;">
_Find songs that match your mood by adjusting the features below. Our AI model will 
recommend songs from a cluster that best fits your preferences._
</div>
""", unsafe_allow_html=True)

st.header("Adjust Your Music Preferences")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        acousticness = st.number_input("Acousticness", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Higher values represent more acoustic-sounding songs.")
        loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-10.0, step=0.1, help="The overall loudness of the track in decibels.")

    with col2:
        instrumentalness = st.number_input("Instrumentalness", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help="Higher values indicate a greater chance of the track having no vocals.")
        energy = st.number_input("Energy", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Measures intensity and activity (e.g., fast, noisy, dynamic).")

user_input = pd.DataFrame(
    [[acousticness, instrumentalness, loudness, energy]],
    columns=['acousticness', 'instrumentalness', 'loudness', 'energy']
)

st.header("Your Recommendations")

with st.spinner('Searching Spotify for your perfect songs...'):
    recommended_songs = recommend_songs(user_input)

if recommended_songs:
    num_cols = 2
    cols = st.columns(num_cols)
    for i, song in enumerate(recommended_songs):
        track_name = song['name']
        artist_name = song['artists'][0]['name']
        album_name = song['album']['name']
        album_art = song['album']['images'][0]['url']
        spotify_preview_url = song.get('preview_url')
        spotify_url = song['external_urls']['spotify']

        # Get the YouTube video URL using the new API function


        with cols[i % num_cols]:
            st.markdown(
                f"""
                <div class="card-container">
                    <img src="{album_art}" alt="Album Art" class="card-img">
                    <div class="card-body">
                        <p class="card-title">{track_name}</p>
                        <p class="card-text">By: {artist_name}</p>
                        <p class="card-text">Album: <em>{album_name}</em></p>
                """,
                unsafe_allow_html=True
            )
            
            if spotify_preview_url:
                with cols[i % num_cols]:
                    st.audio(spotify_preview_url, format="audio/mp3", start_time=0)
            

           

            with cols[i % num_cols]:
                st.markdown(
                    f"""
                        <a href="{spotify_url}" class="spotify-link" target="_blank">
                            Listen on Spotify
                        </a>
                    </div>
                </div>
                    """,
                    unsafe_allow_html=True
                )
else:
    st.write("No recommendations found. Please try again with different preferences.")
