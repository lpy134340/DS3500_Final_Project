from typing import List

import pandas as pd


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    """
    self.song_data = pd.DataFrame(columns=["Song", "Artist", "Genre", "Category", "Lyrics"])
    self.lyric_ngram_data = pd.DataFrame(columns=["Song", "1-Gram", "2-Gram", "3-Gram"])

    def load_song(self, song_name: str) -> None:
        # load song info from spotify, lyrics from genius
        # add info and lyrics to self.song_data
        # generate n-grams, and add to self.lyric_ngram_data
        lyrics = ["the", "song", "lyrics"]
        ngrams = self.generate_ngrams(lyrics)

    def generate_ngrams(self, lyrics: List[str]) -> List[List[str]]:
        output = [["the"], ["the", "song"], ["the", "song", "lyrics"]]
        return output
