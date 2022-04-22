#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:34:58 2022

@author: sarahcosta
"""

import lyricsgenius as lg
import json
import pandas as pd
import time
import socket

def get_song_lyrics(spotify_df, genius):
    
    lyrics_list = []
    for i, song_title in enumerate(spotify_df["name"]):
        artist = list(spotify_df["artists"].values)[i][0]
        #print(artist)
        #print(song_title)
        try:
            song = genius.search_song(song_title, artist)
        except socket.timeout:
            print("next")
            continue
        if song == None:
            lyrics = ""
        else:
            lyrics = song.lyrics
            lyrics = lyrics.split("Lyrics")[1:]
            lyrics = "lyrics".join(lyrics)

        lyrics_list.append(lyrics)
        if i % 5 == 0:
            print(i)
        time.sleep(.5)

    print(lyrics_list)
    spotify_df["lyrics"] = lyrics_list

    return spotify_df


# def main():
#     # connecting to the genius api
#     token = "KErz8_FNBEiaSi6HblkrMIjAs58IhVncRQoN-OGs1Z_AFQ-CzF_juas44Tkup-3u"
#     genius = lg.Genius(token)
#     spotify = "spotify_sample.csv"
#
#     song_df = get_song_lyrics(spotify, genius)
#     song_df.to_csv("lyrics_sample.csv")
#
#
# main()
        


