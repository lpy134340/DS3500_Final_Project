import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
from tqdm import tqdm

with open('client_secrets.json') as f:
    secrets = json.load(f)

os.environ['SPOTIPY_CLIENT_ID'] = secrets["Client_ID"]
os.environ['SPOTIPY_CLIENT_SECRET'] = secrets["Secret"]
os.environ['SPOTIPY_REDIRECT_URI'] = secrets["URL"]

def main():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    print(sp.recommendation_genre_seeds())


if __name__ == "__main__":
    main()
