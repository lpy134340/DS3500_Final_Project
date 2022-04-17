from typing import List

import pandas as pd
import re
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    """
    # columns = ['name', 'artists', 'genre', 'spotifyID', 'lyrics']
    song_data = pd.read_csv("lyrics_sample.csv").drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True)

    # TODO: data cleaning, things other than the NAs, e.g. punctuation marks, remove the chorus stuff?

    lyrics = song_data['lyrics']

    def clean_lyrics(self) -> None:
        """ Remove lyric annotations such as [Chorus] [Verse 1]..
            Remove punctuation
            Remove "Embed" from end of every song
            Remove digits
            Clean lyrics, separate lyrics from one string to a list of strings.
        """
        # NB: nltk requires download of 'stopwords' 'wordnet' and 'omw-1.4'
        # can be downloaded with: python -m nltk.downloader omw-1.4
        # python -m nltk.downloader stopwords
        # python -m nltk.downloader wordnet
        lemmatizer = WordNetLemmatizer()
        for index in range(len(self.song_data)):
            song = self.song_data.iloc[index]
            song['lyrics'] = re.sub("[\(\[].*?[\)\]]", "", song['lyrics'])
            song['lyrics'] = song['lyrics'].translate(str.maketrans('', '', string.punctuation))
            song['lyrics'] = song['lyrics'].replace("Embed", "")
            song['lyrics'] = song['lyrics'].translate(str.maketrans('', '', string.digits))
            # TODO: do we want lyrics as one long string or as list of strings
            # song['lyrics'] = song['lyrics'].split()
            # TODO: Lemmatize works if lyrics are a list of strings
            # song['lyrics'] = [lemmatizer.lemmatize(word) for word in song['lyrics']]

    def remove_stop_words(self) -> None:
        """Load a list of stopwords and remove said stopwords from lyrics data"""
        # TODO: works if song lyrics is a list of strings, but not if one long string
        stop_words = stopwords.words('english')
        for index in range(len(self.song_data)):
            song = self.song_data.iloc[index]
            song["lyrics"] = [word for word in song["lyrics"] if word not in stop_words]

    # create weights df for all the lyrics
    def tfidf(self):
        # TODO: someone else double check if this is right
        vect = TfidfVectorizer(ngram_range=(1,3))
        X = vect.fit_transform(self.lyrics)
        print(vect.get_feature_names_out())
        print(X.shape)

        # get the weights
        lyrics_weights = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
        lyrics_weights["lyrics"] = self.lyrics
        lyrics_weights.set_index("lyrics", inplace=True)
        print(lyrics_weights)
        # print(vect.get_feature_names_out())

    def ML_pipeline(self, models):
        """
        outcome -> genere
        lyrics
            -> n-grams
            -> tfidf
        basic classification one
        """
        # TODO: https://www.kaggle.com/code/neerajmohan/nlp-text-classification-using-tf-idf-features/notebook
        # data loading

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=)
        X_train, X_test, y_train, y_test = train_test_split(self.lyrics, self.song_data['genre'], test_size=.2, random_state = 123)

        # run ML
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))
        tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

        classifier = RandomForestClassifier()

        classifier.fit(tfidf_train_vectors, y_train)

        # evaluate
        y_pred = classifier.predict(tfidf_test_vectors)
        print(classification_report(y_pred, y_test))

        return

    # most influential feature
    def ngram_feature_extraction(self, genre, category):
        return


if __name__ == "__main__":
    library = LyricLibrary()
    library.clean_lyrics()
    #library.remove_stop_words()

    library.tfidf()

    library.ML_pipeline("models")
    # print(library.song_data)
    # print(library.lyrics)

    # print(library.lyrics[0])
    # n_grams = library.generate_ngrams(library.lyrics)
    # print(n_grams)


    # run tf-idf
    # tfidf = library.tfidf()
