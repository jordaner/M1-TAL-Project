from halo import Halo
import numpy as np

class Vectors():
    """
    Class for representing word vectors
    """

    def __init__(self, filepath):
        self.vectors = {}
        with open(filepath, 'r') as file:
            # Skip first line as it does not contain word vector information
            next(file)
            with Halo(text=f"Reading vector data from {filepath}", spinner='dots') as spinner:
                for line in file:
                    split_line = line.split()
                    word = split_line[0]
                    vec = split_line[1:]
                    if word not in self.vectors:
                        self.vectors[word] = np.array([float(x) for x in vec])
                    else:
                        print(f"ERROR : Duplicate vector for word {word}")
            spinner.succeed(f"Finished reading vector data from {filepath}")

    def vector(self, word):
        """
        Returns the vector representation for word
        """
        return self.vectors[word]

    def vocabulary(self):
        """
        Returns the set of words represented
        """
        return {word for word in self.vectors}

    def size(self):
        return len(self.vectors)

    def __getitem__(self, word):
        """
        Returns the vector representation for word
        """
        return self.vector(word)

    def __setitem__(self, word, vector):
        """
        Sets the vector representation for word
        """
        self.vectors[word] = vector


def make_vectors(vectors_filepath):
    """
    Returns a dictionary {string:np.array} containing the words and their corresponding 
    vector representations.
    """
    vectors = {}
    with open(vectors_filepath, 'r') as file:
        # Skip first line as it does not contain word vector information
        next(file)
        with Halo(text=f"Reading vector data from {vectors_filepath}", spinner='dots') as spinner:
            for line in file:
                split_line = line.split()
                word = split_line[0]
                vec = split_line[1:]
                if word not in vectors:
                    vectors[word] = np.array([float(x) for x in vec])
                else:
                    print(f"ERROR : Duplicate vector for word {word}")
        spinner.succeed(f"Finished reading vector data from {vectors_filepath}")
    return vectors

def make_new_vectors(old_vectors):
    # Make new vecs for algorithm, only interested in those vectors that have synonyms 
    # (Maybe even only those that have AT LEAST one with synonym with AT LEAST one
    # vector representation) otherwise no operations will be performed by the algorithm 
    # (hence no improvement can be expected). Although we could end up with a synonym 
    # that has no synonyms not being added when in reality we want it.
    # TODO Come back to this:
    """for word in old_vectors:
        if word in syns:
            for synonym in syns[word]:
                if synonym in old_vectors:
                    new_vecs[word] = np.array(old_vectors[word])"""
    pass



def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_a))