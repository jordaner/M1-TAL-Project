# Retrofitting word vectors

This code is an implementation of Faruqui et al.'s 2015 paper on improving word vector representations by using lexical data.

## Running the code

The code can be run using the following command:
```
python3 main.py
```
By default it will run on the English word vector data and lexical data then perform an evaluation.
There are a number of arguments that can be used to configure the program:
- `-h` shows a help message explaining these arguments
- `-v` used to define the file from which to read the word vectors
- `-l` used to define the file from which to read the lexical data
- `-t` used to define the file from which to read the evaluation data for word pair similarity.
- `-e` used to define the number of epochs hyperparameter.
- `-w` used to run the algorithm in a waterfall fashion (i.e. if epochs=3 run 3 times: with 1, 2 and 3 epochs respectively).
- `-s` used to run the evaluation with sentiment analysis. Not implemented for French.

### Running with French data

A script (french.sh) is available for running the code on the French data. It can be executed using:
```
./french.sh
```


## The data

The datasets used are found in each of their respective folders:
```
data/Embeddings
data/Evaluation
data/Lexical_info
```

### Embedding data

For English we have
``` 
data/Embeddings/vectors_datatxt_250_sg_w10_i5_c500_gensim_clean
```
For French we have
``` 
data/Embeddings/vecs50-linear-frwiki
data/Embeddings/vecs100-linear-frwiki
```

### Evaluation data

For English

```
data/Evaluation/ws353.txt
data/Evaluation/stanford_sentiment_analysis/stanford_raw_test.txt
data/Evaluation/stanford_sentiment_analysis/stanford_raw_train.txt
```

For French

```
data/Evaluation/rg65_french.txt
```

### Lexical data

For English

```
data/Lexical_info/ppdb-2.0-s-lexical
data/Lexical_info/ppdb-2.0-xxxl-lexical
```

For French

```
data/Lexical_info/wolf-1.0b4.xml
```