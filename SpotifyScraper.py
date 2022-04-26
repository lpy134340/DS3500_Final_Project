import spotipy
from spotipy.oauth2 import SpotifyOAuth
import json
import os
import pandas as pd
from tqdm import tqdm

general_genres = {"acoustic": ["acoustic", "guitar", "piano"],
                  "rock": ["alt-rock", "alternative", "emo", "goth", "grunge", "hard-rock", "industrial",
                           "psych-rock", "punk", "punk-rock", "rock", "rock-n-roll"],
                  "metal": ["black-metal", "death-metal", "heavy-metal", "metal", "metal-misc", "metalcore"],
                  "country": ["bluegrass", "country", "folk", "honky-tonk", "rockabilly"],
                  "jazz": ["blues", "jazz", "groove"],
                  "world-music": ["british", "cantopop", "french", "german", "indian", "iranian", "k-pop", "malay", "mandopop",
                                  "philippines-opm", "swedish", "turkish", "world-music"],
                  "brazil": ["bossanova", "brazil", "forro", "mpb", "pagode", "samba", "sertanejo"],
                  "japanese": ["j-dance", "j-idol", "j-pop", "j-rock"],
                  "spanish": ["spanish", "latin", "latino", "salsa", "tango"],
                  "electronic": ["breakbeat", "chicago-house", "club", "dance", "deep-house", "detroit-techno",
                                 "disco", "drum-and-bass", "dub", "dubstep", "edm", "electro", "electronic",
                                 "garage", "grindcore", "hardcore", "hardstyle", "house", "idm", "minimal-techno",
                                 "party", "post-dubstep", "progressive-house", "techno", "trance"],
                  "classical": ["classical", "opera"],
                  "reggae": ["dancehall", "reggae", "reggaeton", "ska"],
                  "hip-hop": ["hip-hop", "trip-hop"],
                  "pop": ["indie", "indie-pop", "new-age", "pop", "pop-film", "power-pop", "synth-pop"],
                  "soul": ["soul", "gospel", "afrobeat"],
                  "r-n-b": ["funk", "r-n-b"],
                  "children": ["children", "disney", "kids"],

                  }
# removed genres
# "misc": ["ambient", "anime", "chill", "comedy", "happy", "holidays", "movies", "new-release",
#                               "rainy-day", "road-trip", "romance", "sad", "show-tunes", "singer-songwriter", "sleep",
#                               "songwriter", "soundtracks", "study", "summer", "work-out"]

def get_specific_to_general():
    """Compute the inverse of the general_genre dictionary"""
    output = {}
    for key in general_genres.keys():
        for value in general_genres[key]:
            output[value] = key
    return output

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
    original_genres = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient',
                       'anime', 'black-metal', 'bluegrass', 'blues', 'bossanova', 'brazil',
                       'breakbeat', 'british', 'cantopop', 'chicago-house', 'children',
                       'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall',
                       'death-metal', 'deep-house', 'detroit-techno', 'disco', 'disney', 'drum-and-bass',
                       'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french',
                       'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'happy',
                       'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'holidays', 'honky-tonk',
                       'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial', 'iranian', 'j-dance', 'j-idol',
                       'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino', 'malay', 'mandopop', 'metal',
                       'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'new-release',
                       'opera', 'pagode', 'party', 'philippines-opm', 'piano', 'pop', 'pop-film', 'post-dubstep',
                       'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'r-n-b', 'rainy-day',
                       'reggae', 'reggaeton', 'road-trip', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'sad',
                       'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter',
                       'soul', 'soundtracks', 'spanish', 'study', 'summer', 'swedish', 'synth-pop', 'tango', 'techno',
                       'trance', 'trip-hop', 'turkish', 'work-out', 'world-music']

    consolidated = ['acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient','anime', 'black-metal', 'bluegrass',
                    'blues', 'brazil', 'breakbeat', 'british', 'children', 'chill', 'classical', 'club', 'comedy',
                    'country', 'dance', 'death-metal', 'disco', 'disney', 'drum-and-bass', 'dub', 'dubstep', 'edm',
                    'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel',
                    'goth', 'grindcore', 'groove', 'grunge', 'guitar', 'hard-rock', 'hardcore', 'hardstyle',
                    'heavy-metal', 'hip-hop', 'holidays', 'house', 'idm', 'indian', 'indie', 'indie-pop', 'industrial',
                    'j-dance', 'j-idol', 'j-pop', 'j-rock', 'jazz', 'k-pop', 'kids', 'latin', 'latino',  'metal',
                    'metal-misc', 'metalcore', 'minimal-techno', 'movies', 'mpb', 'new-age', 'opera', 'party', 'pop',
                    'pop-film', 'post-dubstep', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock',
                    'r-n-b', 'rock', 'rock-n-roll', 'rockabilly', 'romance', 'salsa', 'samba', 'show-tunes', 'soul',
                    'spanish', 'swedish', 'synth-pop', 'tango', 'techno', 'trance', 'turkish', 'world-music']

    removed = ['bossanova', 'cantopop', 'chicago-house', 'dancehall', 'deep-house', 'detroit-techno', 'honky-tonk',
               'iranian', 'new-release', 'pagode', 'philippines-opm', 'piano', 'rainy-day', 'work-out', 'trip-hop',
               'study', 'summer', 'soundtracks', 'road-trip', 'reggae', 'reggaeton', 'happy','sad', 'sertanejo',
               'singer-songwriter', 'ska', 'sleep', 'songwriter', 'malay', 'mandopop',]
    other_languages = []


    # TODO: should we remove other languages too? or consolidate as "world-music"

    loadSpotifyKeys('client_secrets.json')
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth())
    genres = sp.recommendation_genre_seeds()
    df = getSpotifyTracks(sp, genres)
    df.to_csv("data/spotifyTracks.csv")

if __name__ == "__main__":
    main()
