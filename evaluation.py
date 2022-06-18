from scipy import stats

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
            # print(f"{pair}, cosine similarity: {cos_sim}")
            cosine_simalarities.append(cos_sim)
            complete_pairs.append(pair.similarity_score)
            complete_data += 1
        else:
            missing_data += 1
    """print(f"Total complete data = {complete_data}")
    print(f"Total missing data = {missing_data}")"""
    #print(f"Spearman correlation = {stats.spearmanr(cosine_simalarities, complete_pairs)}")
    return stats.spearmanr(cosine_simalarities, complete_pairs).correlation

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