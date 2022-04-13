import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import pandas as pd
from tqdm import tqdm

def loadSpotifyKeys(secrets):
    with open(secrets) as f:
        secrets = json.load(f)

    os.environ['SPOTIPY_CLIENT_ID'] = secrets["Client_ID"]
    os.environ['SPOTIPY_CLIENT_SECRET'] = secrets["Secret"]
    os.environ['SPOTIPY_REDIRECT_URI'] = secrets["URL"]

def getSpotifyTracks(sp, genres, perGenre=20):
    track_dict = {}
    n=0
    for genre in tqdm(genres["genres"]):
        recs = sp.recommendations(seed_genres=[genre], limit=perGenre)
        for track in recs['tracks']:
            track_dict[n] = {"name":track['name'],
                             "artists":[],
                             "genre":genre,
                             "spotifyID":track['id']}
            for artist in track['artists']:
                track_dict[n]["artists"].append(artist['name'])
            n+=1

    df = pd.DataFrame(track_dict).T
    return df

def main():
    loadSpotifyKeys('client_secrets.json')
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    genres = sp.recommendation_genre_seeds()
    df = getSpotifyTracks(sp, genres)
    df.to_csv("data/spotifyTracks.csv")

if __name__ == "__main__":
    main()
