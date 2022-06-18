from collections import defaultdict
from halo import Halo
from my_frenetic import *

ENTAILMENTS = {"Equivalence", "OtherRelated"}

class Lexicon():
    """
    Class for representing lexicon with synonym, hypernym, antonym relations
    """

    def __init__(self, filepath, filetype, vocabulary=None):
        """
        PARAMS 
        filepath : Absolute path to file containing lexical data
        filetype : Gives the type of the file, compatible files above:
        {WOLF, PPDB}
        """
        self.COMPATIBLE_FILETYPES = {"WOLF", "PPDB"}
        self.filepath = filepath
        self.synonyms = defaultdict(set)
        self._vocabulary = vocabulary
        if filetype not in self.COMPATIBLE_FILETYPES:
            print(f"ERROR : Incompatible filetype: {filetype}. Compatible filetypes are: {self.COMPATIBLE_FILETYPES}")
        self.filetype = filetype

    def read(self):
        """
        Calls the appropriate read function based on filetype.
        All read function instantiate the lexicon dictionary {string:[string]} with the 
        key representing a word and the value being the list of all synonyms for that 
        word.
        """
        if self.filetype == "PPDB":
            self.read_ppdb()
        elif self.filetype == "WOLF":
            self.read_wolf()
        else:
            print(f"ERROR : Incompatible filetype: {filetype}. Compatible filetypes are: {self.COMPATIBLE_FILETYPES}")

    def read_ppdb(self):
        """
        Reads a file in the ppdb 2.0 format
        """
        with open(self.filepath, 'r') as file:
            with Halo(text=f"Reading synonyms from {self.filepath}", spinner='dots') as spinner:
                for line in file:
                    split_line = line.split(' ||| ')
                    current_word = split_line[1].strip()
                    current_paraphrase = split_line[2].strip()
                    current_entailment = split_line[-1].strip()
                    if current_entailment in ENTAILMENTS:
                        self.synonyms[current_word].append(current_paraphrase)
                spinner.succeed(f"Completed {self.filepath}")

    def read_wolf(self):
        """
        Reads a file in the WOLF xml format
        """
        with Halo(text=f"Reading synonyms from {self.filepath}", spinner='dots') as spinner:
            lexicon = FreNetic(self.filepath)
            for word in self._vocabulary:
                synsets = lexicon.synsets(word)
                if synsets:
                    for synset in synsets:
                        for synonym in synset.literals():
                            if synonym.span() in self._vocabulary:
                                self.synonyms[word].add(synonym.span())
                    self.synonyms[word].discard(word)
            spinner.succeed(f"Completed {self.filepath}")

    def vocabulary(self):
        if self._vocabulary:
            return self._vocabulary
        else:
            return {word for word in self.synonyms}

    def get_synonyms(self, word):
        """
        Returns the set of synonyms for a given word
        """
        return self.synonyms[word]

    def __getitem__(self, word):
        return self.get_synonyms(word)

def make_synonyms(lexicon_filepath):
    """
    Returns a dictionary {string:[string]} with the key representing a word and the 
    value being the list of all synonyms for that word.
    """
    synonyms = defaultdict(list)

    with open(lexicon_filepath, 'r') as file:
        lines = []
        if False: # DEBUG
            lines = [next(file) for x in range(100)]
        else:
            with Halo(text=f"Reading synonyms from {lexicon_filepath}", spinner='dots') as spinner:
                for line in file:
                    split_line = line.split(' ||| ')
                    current_word = split_line[1].strip()
                    current_paraphrase = split_line[2].strip()
                    current_entailment = split_line[-1].strip()
                    if current_entailment in ENTAILMENTS:
                        synonyms[current_word].append(current_paraphrase)
                spinner.succeed(f"Completed {lexicon_filepath}")
    return synonyms

def make_synonyms_from_frenetic(filepath, vocabulary):
    with Halo(text=f"Reading synonyms from {filepath}", spinner='dots') as spinner:
        lexicon = FreNetic(filepath)
        synonyms = defaultdict(set)
        for word in vocabulary:
            synsets = lexicon.synsets(word)
            if synsets:
                for synset in synsets:
                    for synonym in synset.literals():
                        if synonym.span() in vocabulary:
                            synonyms[word].add(synonym.span())
                synonyms[word].discard(word)
        spinner.succeed(f"Completed {filepath}")
    return synonyms