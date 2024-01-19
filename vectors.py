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
        if word in self.vectors:
            return self.vectors[word]
        else:
            return None

    def vocabulary(self):
        """
        Returns the set of words represented
        """
        return {word for word in self.vectors}

    def size(self):
        return len(self.vectors)

    def dimension(self):
        """
        Returns the dimension of the vectors
        """
        for word in self.vectors:
            return len(self.vectors[word])

        return 0

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

def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_a))