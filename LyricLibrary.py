from typing import List

import pandas as pd


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    """
    song_data = pd.DataFrame(columns=["Song", "Artist", "Genre", "Category", "Lyrics"])
    lyric_ngram_data = pd.DataFrame(columns=["Song", "1-Gram", "2-Gram", "3-Gram"])

    def load_song(self, song_name: str) -> None:
        # load song info from spotify, lyrics from genius
        # add info and lyrics to self.song_data
        # generate n-grams, and add to self.lyric_ngram_data
        lyrics = ["the", "song", "lyrics"]
        ngrams = self.generate_ngrams(lyrics)

    def generate_ngrams(self, lyrics: List[str]) -> List[List[str]]:
        n_size = 5
        output = []

        # loop through every index in lyrics
        for index in range(len(lyrics)):
            current = []
            # add a 1-gram starting at lyrics[index], then 2-gram, then repeat til n
            for n in range(n_size):
                if index + n >= len(lyrics):
                    continue
                current.append(lyrics[index + n])
                copy_current = current.copy()
                output.append(copy_current)
        return output


if __name__ == "__main__":
    library = LyricLibrary()
    lyrics = ["the", "other", "day", "I", "did", "a", "thing", "and", "it", "was", "cool"]
    n_grams = library.generate_ngrams(lyrics)
    print(n_grams)
