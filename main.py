from argparse import ArgumentParser
from copy import deepcopy
from evaluation import SimilarityPair
from halo import Halo
import os
from tqdm import tqdm

from vectors import *
from lexicon import *
from evaluation import *

DEBUG = False

DATA_FOLDER = "data/"
ENGLISH_VECTORS = "Embeddings/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean"
#ENGLISH_LEXICON_FILES = ["Lexical_info/ppdb-2.0-xxxl-lexical"]
ENGLISH_LEXICON_FILES = ["Lexical_info/ppdb-2.0-s-lexical"]
ENGLISH_EVALUATION = "Evaluation/ws353.txt"
POSSIBLE_LEXICONS = {"s", "l", "xxxl"}

def retrofit(lexicon, old_vectors, epochs=10):
    """
    Per epoch:
    (for each synonym Beta * new_vector + Alpha * old_vector) / Beta * num(synonyms) + Alpha 
    (for each synonym 1/degree * new_vector + 1 * old_vector) / num(synonyms) + 1 
    """
    new_vectors = deepcopy(old_vectors)
    loop_vocab = new_vectors.vocabulary().intersection(lexicon.vocabulary())
    word_num = 0
    for e in range(epochs):
        print(f"Epoch: {e}")
        with tqdm(total=len(loop_vocab)) as pbar:
            for word in loop_vocab:
                #adjacent_words = set(lexicon[word]).intersection(new_vectors.vocabulary())
                adjacent_words = lexicon[word]
                alpha = 1
                if len(adjacent_words) == 0:
                    pbar.update(1)
                    continue
                beta = 1/len(adjacent_words)

                updated_vector = old_vectors[word] * alpha

                for adjacent_word in adjacent_words:
                    updated_vector += new_vectors[adjacent_word] * beta

                new_vectors[word] = updated_vector/2
                pbar.update(1)
    return new_vectors

def old_retrofit(lexicon, old_vectors, epochs=10):
    """
    Per epoch:
    (for each synonym Beta * new_vector + Alpha * old_vector) / Beta * num(synonyms) + Alpha 
    (for each synonym 1/degree * new_vector + 1 * old_vector) / num(synonyms) + 1 
    """
    new_vectors = deepcopy(old_vectors)
    loop_vocab = new_vectors.vocabulary().intersection(set(lexicon))
    word_num = 0
    for e in range(epochs):
        print(f"Epoch: {e}")
        with tqdm(total=len(loop_vocab)) as pbar:
            for word in loop_vocab:
                #adjacent_words = set(lexicon[word]).intersection(new_vectors.vocabulary())
                adjacent_words = lexicon[word]
                alpha = 1
                if len(adjacent_words) == 0:
                    pbar.update(1)
                    continue
                beta = 1/len(adjacent_words)

                updated_vector = old_vectors[word] * alpha

                for adjacent_word in adjacent_words:
                    updated_vector += new_vectors[adjacent_word] * beta

                new_vectors[word] = updated_vector/2
                pbar.update(1)
    return new_vectors



def calculate_overlap(vectors, synonyms):
    vecs_with_synonyms = 0
    synonyms_with_vec = 0
    for word in vectors:
        if word in synonyms:
            vecs_with_synonyms += 1
            for synonym in synonyms[word]:
                synonyms_with_vec += 1
            if DEBUG:
                print(f"Synonyms for {word}: {synonyms[word]}")

    return vecs_with_synonyms, synonyms_with_vec

def show_all_synonyms(all_synonyms):
    for word in all_synonyms:
        synonyms = all_synonyms[word]
        if len(synonyms):
            print(f"Synonyms for {word}: \n\t{synonyms}\n")

def show_synonyms(synonyms, word):
    """
    Prints out the list of synonyms for a given word, if there are any.
    """
    if word in synonyms:
        print(f"Synonyms for {word}: {synonyms[word]}")
    else:
        print(f"No synonyms for {word}")

def show_vector(vectors, word):
    """
    Prints out the word vector for the given word if one exists
    """
    if word in vectors:
        print(f"Vector for {word}: {vectors[word]}")
    else:
        print(f"No vector for {word}")

def show_results(lexicon, vectors, epochs, correlation):
    print(f"Results for:\nLexicon: {lexicon}\nVectors: {vectors}\nEpochs: {epochs}\nCorrelation: {correlation}")

def do_english():
    old_vectors = make_vectors(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.vectors))
    if args.lexicon in POSSIBLE_LEXICONS:
        synonyms = make_synonyms("data/Lexical_info/ppdb-2.0-" + args.lexicon + "-lexical")
    else:
        print(f"ERROR : Invalid lexicon, the valid lexicons are {POSSIBLE_LEXICONS}")
        exit()
    
    # See make_new_vectors above
    # 11/06 15h, just do basic version below come back later if time permits.

    similarity_pairs = make_similarity_pairs(os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FOLDER + ENGLISH_EVALUATION), "\t")

    correls = []

    for e in range(1, args.epochs):
        new_vectors = retrofit(synonyms, old_vectors, e)
        correls.append(evaluate(new_vectors, similarity_pairs))
        """
    new_vectors = retrofit(synonyms, old_vectors, args.epochs)

    evaluate(old_vectors, similarity_pairs)
    correl = evaluate(new_vectors, similarity_pairs)

    show_results(args.lexicon, args.vectors, args.epochs, correl)"""
    for e in range(1, args.epochs):
        show_results(args.lexicon, args.vectors, e, correls[e-1])


# MAIN

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--debug", dest="DEBUG", type=bool, default=False,
                        help="Run program in debug mode (False by default)")
    parser.add_argument("-r", "--read", dest="read", type=bool, default=True,
                        help="Read data for algo (True by default), set to False for "+
                        "testing purposes")
    parser.add_argument("-v", "--vectors", dest="vectors", type=str, 
                        default="data/Embeddings/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean",
                        help="Define the relative path to the vectors to use for retrofitting.")
    parser.add_argument("-l", "--lexicon", dest="lexicon", type=str, default="s",
                        help="Define the lexicon to use for retrofitting.")
    parser.add_argument("-t", "--test", dest="test", type=str, 
                        default="data/Evaluation/ws353.txt",
                        help="Define the relative path to the evaluation data.")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, 
                        default=10,
                        help="The number of epochs to run the retrofitting algorithm.")

    args = parser.parse_args()
    DEBUG = args.DEBUG
    
    ancien_vecs = Vectors(args.vectors)
    lexicon_fr = Lexicon(args.lexicon, "WOLF", ancien_vecs.vocabulary())
    lexicon_fr.read()
    #lexicon_fr = make_synonyms_from_frenetic(args.lexicon, ancien_vecs.vocabulary())

    similarity_pairs = make_similarity_pairs(args.test)
    new_vectors = retrofit(lexicon_fr, ancien_vecs, args.epochs)
    #new_vectors = old_retrofit(lexicon_fr, ancien_vecs, args.epochs)

    evaluate(ancien_vecs, similarity_pairs)
    old_correl = evaluate(ancien_vecs, similarity_pairs)
    new_correl = evaluate(new_vectors, similarity_pairs)

    show_results(args.lexicon, args.vectors, 0, old_correl)
    show_results(args.lexicon, args.vectors, args.epochs, new_correl)