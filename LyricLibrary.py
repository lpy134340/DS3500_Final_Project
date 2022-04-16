from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    """
    # song_data = pd.DataFrame(columns=["Song", "Artist", "Genre", "Category", "Lyrics"])
    song_data = pd.read_csv("lyrics_sample.csv").drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True)

    lyric_ngram_data = pd.DataFrame(columns=["Song", "1-Gram", "2-Gram", "3-Gram"])
    # TODO: data cleaning, things other than the NAs, e.g. punctuation marks, remove the chorus stuff?

    lyrics = song_data['lyrics']

    # TODO: do we still need this method?
    def load_song(self, song_name: str) -> None:
        # load song info from spotify, lyrics from genius
        # add info and lyrics to self.song_data
        # generate n-grams, and add to self.lyric_ngram_data
        lyrics = ["the", "song", "lyrics"]
        ngrams = self.generate_ngrams(lyrics)

    def generate_ngrams(self, lyrics: List[str]) -> List[List[str]]:
        """ Create a list of n-grams for one song.
        lyrics: a list of strings representing the lyrics in one song
        returns: the n-grams, represented by a list of list of strings
        Example:
            ["I" "am" "cool"] -> [["I"], ["I", "am"], ["I", "am", "cool"], ["am"], ["am", "cool"], ["cool"]]
            """
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

    # create weights df for all the lyrics
    def tfidf(self):
        # TODO: someone else double check if this is right
        vect = TfidfVectorizer()
        X = vect.fit_transform(self.lyrics)
        print(vect.get_feature_names_out())
        print(X.shape)

        # get the weights
        lyrics_weights = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
        lyrics_weights["lyrics"] = self.lyrics
        lyrics_weights.set_index("lyrics", inplace=True)
        print(lyrics_weights)
        #print(vect.get_feature_names_out())

    def ML_pipeline(self, models):
        """
        outcome -> genere
        lyrics
            -> n-grams
            -> tfidf
        basic classification one
        """
        # data loading

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=)

        # run ML

        # evaluate

        return

    # most influential feature
    def ngram_feature_extraction(self, genre, category):
        return


if __name__ == "__main__":

    library = LyricLibrary()
    #print(library.song_data)
    #print(library.lyrics)

    # run n-grams
    # lyrics = ["the", "other", "day", "I", "did", "a", "thing", "and", "it", "was", "cool"]
    print(library.lyrics[0])
    n_grams = library.generate_ngrams(library.lyrics)
    # print(n_grams)

    # run tf-idf
    # tfidf = library.tfidf()