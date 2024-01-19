from halo import Halo
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from vectors import cosine_similarity


def make_similarity_pairs(similarity_filepath, seperator=" "):
    similarity_pairs = []
    with open(similarity_filepath, 'r') as file:
        for line in file:
            split_line = line.split(seperator)
            current_similarity_pair = SimilarityPair(split_line[0], split_line[1], float(split_line[2].strip()))
            similarity_pairs.append(current_similarity_pair)
    return similarity_pairs

def evaluate(vectors, similarity_pairs):
    missing_data = 0
    complete_data = 0
    cosine_simalarities = []
    complete_pairs = []
    for pair in similarity_pairs:
        if pair.first_word in vectors.vocabulary() and pair.second_word in vectors.vocabulary():
            first_vec = vectors[pair.first_word]
            second_vec = vectors[pair.second_word]
            cos_sim = cosine_similarity(first_vec, second_vec)
            cosine_simalarities.append(cos_sim)
            complete_pairs.append(pair.similarity_score)
            complete_data += 1
        else:
            missing_data += 1
    return stats.spearmanr(cosine_simalarities, complete_pairs).correlation

def vectorize_from_path(filepath, vectors):
    y = []
    X = []
    with Halo(text=f"Reading review data from {filepath}", spinner='dots') as spinner:
        with open(filepath, 'r') as file:
            for line in file:
                split_line = line.split()
                current_y = int(split_line[0])
                x_words = split_line[1:]
                x_vec = np.zeros(vectors.dimension())
                for word in x_words:
                    if vectors[word] is not None:
                        x_vec += np.array(vectors[word])
                    else:
                        x_vec += np.zeros(vectors.dimension())
                y.append(current_y)
                X.append(x_vec)
        spinner.succeed(f"Finished reading review data from {filepath}")
    return X, y

def get_accuracy(model, X, y):
    num_samples = len(y)
    correct = 0
    for i in range(num_samples):
        if model.predict(X[i].reshape(1, -1)) == y[i]:
            correct += 1

    return correct/num_samples

def sentiment_analysis(filepath, vectors):
    train_X, train_y = vectorize_from_path("data/Evaluation/stanford_sentiment_analysis/stanford_raw_train.txt", vectors)

    clf = LogisticRegression(random_state=0, max_iter=500).fit(train_X, train_y)

    test_X, test_y = vectorize_from_path("data/Evaluation/stanford_sentiment_analysis/stanford_raw_test.txt", vectors)

    print(f"Train Accuracy: {get_accuracy(clf, train_X, train_y)}")
    print(f"Test Accuracy: {get_accuracy(clf, test_X, test_y)}")

class SimilarityPair:
    """
    Class for containing pairs of words and a similarity rating.
    """

    def __init__(self, first_word, second_word, similarity_score):
        """
        PARAMS
        first_word: string
        second_word: string
        similarity_score: float
        """
        self.first_word = first_word
        self.second_word = second_word
        self.similarity_score = similarity_score

    def __str__(self):
        return self.first_word + ":" + self.second_word + " = " + str(self.similarity_score)