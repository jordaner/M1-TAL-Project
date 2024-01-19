from argparse import ArgumentParser
from copy import deepcopy
from evaluation import SimilarityPair
from halo import Halo
import os
from tqdm import tqdm

from vectors import *
from lexicon import *
from evaluation import *

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
                    if new_vectors[adjacent_word] is not None:
                        updated_vector += new_vectors[adjacent_word] * beta

                new_vectors[word] = updated_vector/2
                pbar.update(1)
    return new_vectors

def show_results(lexicon, vectors, epochs, correlation):
    """
    Prints the results for the Spearman correlation test
    """
    print(f"Results for:\nLexicon: {lexicon}\nVectors: {vectors}\nEpochs: {epochs}\nCorrelation: {correlation}")

def new_english(args):
    pass

def do_french(args):
    ancien_vecs = Vectors(args.vectors)
    lexicon_fr = Lexicon(args.lexicon, "WOLF", ancien_vecs.vocabulary())
    lexicon_fr.read()
    #lexicon_fr = make_synonyms_from_frenetic(args.lexicon, ancien_vecs.vocabulary())

    similarity_pairs = make_similarity_pairs(args.test)
    new_vectors = retrofit(lexicon_fr, ancien_vecs, args.epochs)
    #new_vectors = old_retrofit(lexicon_fr, ancien_vecs, args.epochs)

    old_correl = evaluate(ancien_vecs, similarity_pairs)
    new_correl = evaluate(new_vectors, similarity_pairs)

    show_results(args.lexicon, args.vectors, 0, old_correl)
    show_results(args.lexicon, args.vectors, args.epochs, new_correl)


# MAIN

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--vectors", dest="vectors", type=str, 
                        default="data/Embeddings/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean",
                        help="Define the vectors to use for retrofitting.")
    parser.add_argument("-l", "--lexicon", dest="lexicon", type=str, default="data/Lexical_info/ppdb-2.0-s-lexical",
                        help="Define the lexicon to use for retrofitting.")
    parser.add_argument("-t", "--test", dest="test", type=str, 
                        default="data/Evaluation/ws353.txt",
                        help="Define the evaluation data for word pair similarity.")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, 
                        default=10,
                        help="The number of epochs to run the retrofitting algorithm.")
    parser.add_argument("-w", "--waterfall", dest="waterfall", type=bool, 
                        default=False,
                        help="Run epochs in a waterfall fashion (i.e. if e=3 run 3 times: with 1, 2 and 3 epochs respectively).")
    parser.add_argument("-s", "--sentiment_analysis", dest="sentiment_analysis", type=bool, 
                        default=False,
                        help="Run sentiment evaluation or not, default False. Sentiment analysis currently not available for French.")
    

    args = parser.parse_args()
    
    lexicon_style = ""

    if "ppdb" in args.lexicon:
        lexicon_style = "PPDB"
    elif "wolf" in args.lexicon:
        lexicon_style = "WOLF"
    else:
        print(f"ERROR : Invalid lexicon. Lexicon name must contain wolf or ppdb.")
        exit()

    old_vecs = Vectors(args.vectors)
    vocab = None
    if lexicon_style == "WOLF":
        vocab = old_vecs.vocabulary()

    lexicon = Lexicon(args.lexicon, lexicon_style, vocab)
    lexicon.read()

    similarity_pairs = make_similarity_pairs(args.test)

    if args.waterfall:
        correls = []

        for e in range(1, args.epochs):
            new_vectors = retrofit(lexicon, old_vecs, e)
            correls.append(evaluate(new_vectors, similarity_pairs))
        
        for e in range(1, args.epochs):
            show_results(args.lexicon, args.vectors, e, correls[e-1])
    
    else:
        new_vectors = retrofit(lexicon, old_vecs, args.epochs)

        old_correl = evaluate(old_vecs, similarity_pairs)
        new_correl = evaluate(new_vectors, similarity_pairs)

        show_results(args.lexicon, args.vectors, 0, old_correl)
        show_results(args.lexicon, args.vectors, args.epochs, new_correl)

    if args.sentiment_analysis:
        print("Old sentiment analysis")
        sentiment_analysis("data/Evaluation/stanford_sentiment_analysis/stanford_raw_train.txt", old_vecs)
        print("New sentiment analysis")
        sentiment_analysis("data/Evaluation/stanford_sentiment_analysis/stanford_raw_train.txt", new_vectors)

    #new_english(args)
    #do_french(args)