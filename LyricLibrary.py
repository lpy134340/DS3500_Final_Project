from typing import List

import pandas as pd
import re
import string

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    """
    # columns = ['name', 'artists', 'genre', 'spotifyID', 'lyrics']
    song_data = pd.read_csv("lyrics_200_per.csv").drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True)
    # Accuracy results
    knn_results = pd.DataFrame(columns=["Number Neighbors", "Accuracy"])
    # TODO: Should we use min_sample_split?
    random_forest_results = pd.DataFrame(columns=["Max Depth", "Accuracy"])

    lyrics = song_data['lyrics']

    def clean_lyrics(self) -> None:
        """ Remove lyric annotations such as [Chorus] [Verse 1]..
            Remove punctuation
            Remove "Embed" from end of every song
            Remove digits
            Remove newlines
            Remove stopwords
            Lemmatize
        """
        # NB: nltk requires download of 'stopwords' 'wordnet' and 'omw-1.4'
        # can be downloaded with: python -m nltk.downloader omw-1.4
        # python -m nltk.downloader stopwords
        # python -m nltk.downloader wordnet
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        for index in range(len(self.song_data)):
            song = self.song_data.iloc[index]
            song['lyrics'] = re.sub("[\(\[].*?[\)\]]", "", song['lyrics'])
            song['lyrics'] = song['lyrics'].translate(str.maketrans('', '', string.punctuation))
            song['lyrics'] = song['lyrics'].replace("Embed", "")
            song['lyrics'] = song['lyrics'].translate(str.maketrans('', '', string.digits))
            song['lyrics'] = song['lyrics'].replace("\n", " ")
            # Convert string to list of strings, remove stopwords, and lemmatize
            lyrics_as_list = song['lyrics'].split()
            lyrics_as_list = [word for word in lyrics_as_list if word not in stop_words]
            lyrics_as_list= [lemmatizer.lemmatize(word) for word in lyrics_as_list]
            # convert back from list of strings to string
            song['lyrics'] = " ".join(lyrics_as_list)

    # TODO: Deprecated method, stopwords are removed in clean_lyrics
    def remove_stop_words(self) -> None:
        """Load a list of stopwords and remove said stopwords from lyrics data"""
        stop_words = stopwords.words('english')
        for index in range(len(self.song_data)):
            song = self.song_data.iloc[index]
            song["lyrics"] = [word for word in song["lyrics"] if word not in stop_words]

    # create weights df for all the lyrics
    # TODO: deprecated, tfidf handled in ML_pipeline
    def tfidf(self):
        vect = TfidfVectorizer(ngram_range=(3, 3))
        X = vect.fit_transform(self.lyrics)

        # get the weights
        lyrics_weights = pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())
        lyrics_weights["lyrics"] = self.lyrics
        lyrics_weights.set_index("lyrics", inplace=True)
        # print(vect.get_feature_names_out())

    # SKLearn Classifier -> Double
    def ML_pipeline(self, classifier):
        """
            Trains the data and fits it to a given classifier. Returns the results as a dataframe
            with precision and recall per genre.
        """
        # TODO: https://www.kaggle.com/code/neerajmohan/nlp-text-classification-using-tf-idf-features/notebook

        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=)
        X_train, X_test, y_train, y_test = train_test_split(self.lyrics, self.song_data['genre'], test_size=.2,
                                                            random_state=123)
        # run ML
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test_vectors = tfidf_vectorizer.transform(X_test)
        classifier.fit(tfidf_train_vectors, y_train)

        # evaluate
        y_pred = classifier.predict(tfidf_test_vectors)

        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy * 100

    # most influential feature
    def ngram_feature_extraction(self, genre, category):
        return

    def run_all_models(self, models: dict):
        """Runs a collection of models with different parameters and saves their results"""
        for n in models["knn"]:
            knn = KNeighborsClassifier(n_neighbors=n, metric='cosine')
            score = self.ML_pipeline(knn)
            result = {"Number Neighbors": n, "Accuracy": score}
            # TODO: Suppress Warning for frame.append
            self.knn_results = self.knn_results.append(result, ignore_index=True)
        for depth in models["forest"]:
            random_forest = RandomForestClassifier(max_depth=depth)
            score = self.ML_pipeline(random_forest)
            result = {"Max Depth": depth, "Accuracy": score}
            self.random_forest_results = self.random_forest_results.append(result, ignore_index=True)
        return

    def plot_model_accuracy(self):
        """Plot the accruacy of KNN models with different number of neighbors"""
        xs = self.knn_results["Number Neighbors"]
        y1 = self.knn_results["Accuracy"]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=y1, s=5, color="blue")
        sns.lineplot(x=xs, y=y1, color="blue")

        plt.grid()
        plt.xlim(0, max(xs) * 1.2)

        plt.xlabel("Number of Neighbors")
        plt.ylabel("Accuracy %")
        plt.title("Accuracy of KNN Models Predicting Song Genre by lyric")
        plt.savefig("knn_accuracy.png", bbox_inches='tight')
        plt.show()

    def plot_model_accuracy_forest(self):
        """Plot the accruacy of KNN models with different number of neighbors"""
        xs = self.random_forest_results["Max Depth"]
        y1 = self.random_forest_results["Accuracy"]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=y1, s=5, color="blue")
        sns.lineplot(x=xs, y=y1, color="blue")

        plt.grid()
        plt.xlim(0, max(xs) * 1.2)

        plt.xlabel("Max Depth")
        plt.ylabel("Accuracy %")
        plt.title("Accuracy of Random Forest Models vs Max Depth")
        plt.savefig("forest_accuracy.png", bbox_inches='tight')
        plt.show()



if __name__ == "__main__":
    library = LyricLibrary()
    library.clean_lyrics()

    #library.tfidf()

    #library.ML_pipeline("models")
    # print(library.song_data)
    print(len(library.lyrics))
    print(library.lyrics)

    # print(library.lyrics[0])
    # n_grams = library.generate_ngrams(library.lyrics)
    # print(n_grams)

    sample_models = {
        "knn": [3, 5, 7, 9, 11],
        "forest": [4, 5, 6]
    }
    library.run_all_models(sample_models)
    print(library.knn_results)
    print(library.random_forest_results)

    library.plot_model_accuracy()
    library.plot_model_accuracy_forest()
