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
from SpotifyScraper import general_genres, get_specific_to_general


class LyricLibrary:
    """Lyric Library provides NLP tools for song analysis.
    This class can download song data from spotify and genius.
    The Machine Learning pipeline creates a ML model to predict genre based on lyrics
    """
    # columns = ['name', 'artists', 'genre', 'spotifyID', 'lyrics']
    song_data = pd.read_csv("Data/full_lyric_dataset.csv").drop(columns=['Unnamed: 0']).dropna().reset_index(drop=True)
    # Accuracy results
    knn_results = pd.DataFrame(columns=["Number Neighbors", "Accuracy"])
    random_forest_results = pd.DataFrame(columns=["Max Depth", "Accuracy"])

    lyrics = song_data['lyrics']

    def generalize_genres(self):
        """remove specific genre classifications and replace with generic umbrella genres"""
        specific_to_general = get_specific_to_general()
        for index in range(len(self.song_data)):
            new_genre = specific_to_general[self.song_data.iloc[index]['genre']]
            self.song_data.iloc[index]['genre'] = new_genre

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
            lyrics_as_list = [lemmatizer.lemmatize(word) for word in lyrics_as_list]
            # convert back from list of strings to string
            song['lyrics'] = " ".join(lyrics_as_list)

    # SKLearn Classifier -> Double
    def ML_pipeline(self, classifier):
        """
            Trains the data and fits it to a given classifier. Returns the accuracy as a percent
        """
        # train test split
        # X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=)
        X_train, X_test, y_train, y_test = train_test_split(self.lyrics, self.song_data['genre'], test_size=.2,
                                                            random_state=123)
        # run ML
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
        tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test_vectors = tfidf_vectorizer.transform(X_test)
        classifier.fit(tfidf_train_vectors, y_train)

        # evaluate
        y_pred = classifier.predict(tfidf_test_vectors)

        test_accuracy = accuracy_score(y_test, y_pred)
        return test_accuracy * 100

    def run_all_models(self, models: dict):
        """Runs a collection of models with different parameters and saves their results"""
        if "knn" in models.keys():
            for n in models["knn"]:
                knn = KNeighborsClassifier(n_neighbors=n, metric='cosine')
                score = self.ML_pipeline(knn)
                result = {"Number Neighbors": n, "Accuracy": score}
                self.knn_results = self.knn_results.append(result, ignore_index=True)
        if "forest" in models.keys():
            for depth in models["forest"]:
                random_forest = RandomForestClassifier(max_depth=depth, min_samples_split=2)
                score = self.ML_pipeline(random_forest)
                result = {"Max Depth": depth, "Accuracy": score}
                self.random_forest_results = self.random_forest_results.append(result, ignore_index=True)
        return

    def plot_model_accuracy(self):
        """Plot the accuracy of KNN models with different number of neighbors"""
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
        """Plot the accuracy of KNN models with different number of neighbors"""
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


def main():
    """Create a LyricLibrary, clean its data, run some ML models and plot the accuracy"""
    library = LyricLibrary()

    library.generalize_genres()  # consolidate genres to more general terms
    library.clean_lyrics()  # clean data

    sample_models = {
        "knn": [3, 5, 7, 9, 11, 13, 15, 17, 19],
        "forest": [6, 7, 8, 9, 10, 11, 12]
    }

    library.run_all_models(sample_models)  # run all knn and forest models

    print(library.knn_results)
    print(library.random_forest_results)

    library.plot_model_accuracy()
    library.plot_model_accuracy_forest()


if __name__ == "__main__":
    main()
