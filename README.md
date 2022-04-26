# DS3500 Final Project: Spotify and Genius

We aim to expand on the NLP Library homework. We used and will use the Genius API and a library called LyricsGenius to extract lyrics for different songs and compare both individual songs and albums. There are multiple ways that we can expand on the existing library from homework 2. First, we would like to implement a sentiment analysis on the lyrics. We would also go even further by connecting to the Spotify API. The Spotify API assigns each song a category that they think best describes the song (e.g., energetic, wistful, party, etc.). It also can return the genre of a song. We would like to build a machine learning model that predicts the category and/or genre of a song based on n-gram text feature extraction and tf-idf vectorization. We would also like to see if there are any words that are popular in a song of a particular category. From there we can see what lyrics are tied most to a song’s category/genre. 


The Driver.py script runs all the important code for this project. Driver will call methods in get_song_lyrics.py and LyricLibrary.py to download, clean, and fit data to ML models that predict genre as a result of song lyrics.
