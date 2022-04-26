import spotipy
from spotipy.oauth2 import SpotifyOAuth
from SpotifyScraper import getSpotifyTracks, loadSpotifyKeys
from get_song_lyrics import get_song_lyrics
import lyricsgenius as lg
from api_key import spotify_api_key
from LyricLibrary import main as lyric_library_main
import pandas as pd

if __name__ == "__main__":
    # get spotify api data
    loadSpotifyKeys('client_secrets.json')
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    genres = sp.recommendation_genre_seeds()
    spotify_df = getSpotifyTracks(sp, genres, perGenre=100)

    # get genius api data
    genius = lg.Genius(spotify_api_key, timeout=15, retries=3)
    song_df = get_song_lyrics(spotify_df, genius)
    print(song_df)

    song_df.to_csv("full_lyric_dataset.csv")

    # Run the Lyric Library main method to generate machine learning models
    # lyric_library_main()
