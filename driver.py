import spotipy
from spotipy.oauth2 import SpotifyOAuth
from SpotifyScraper import getSpotifyTracks, loadSpotifyKeys
from get_song_lyrics import get_song_lyrics
import lyricsgenius as lg
from api_key import spotify_api_key
from LyricLibrary import LyricLibrary
import pandas as pd

if __name__ == "__main__":
    # get spotify api data
    loadSpotifyKeys('client_secrets.json')
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    genres = sp.recommendation_genre_seeds()
    spotify_df = getSpotifyTracks(sp, genres, perGenre=1)

    # get genius api data
    genius = lg.Genius(spotify_api_key)
    song_df = get_song_lyrics(spotify_df, genius)
    print(song_df)
    
    # song_df.to_csv("lyrics_sample.csv")

    # library = LyricLibrary()
    # # lyrics = ["the", "other", "day", "I", "did", "a", "thing", "and", "it", "was", "cool"]
    # # n_grams = library.generate_ngrams(lyrics) # generate n-grams for one song?
    # # print(n_grams)
    # tfidf = library.tfidf()