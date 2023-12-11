import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy


## Task 1: Extract word features and show EDA - inspect, visualise and clean data
# Show a few visualizations like histograms. Describe any data cleaning procedures.
# Based on your EDA, what is your plan of analysis?
#
# Look at online resources on processing raw texts to feature vectors. Many methods process texts
# to matrix form (word embedding), including TF-IDF, GloVe, Word2Vec, etc. Pick a method and process the raw texts to
# word embedding. Briefly explain the method(s) and how they work in your own words. Also, do exploratory data analysis
# such as word statistics and/or visualisation.

# load the data
train = pd.read_csv('data/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('data/learn-ai-bbc/BBC News Test.csv')

# take a look at the first few rows of each dataset - notice that the test dataset doesn't include labels so we're not
# going to be able to use it for model testing. Instead, we're going to have to rely on the training dataset.
train.head()
test.head()

# check out the distribution of category labels
# we can see that the distribution is fairly even, which is good, but business and sports articles are slightly more
# common than tech, politics or entertainment articles
sns.countplot(data=train, x="Category", color="seagreen")
plt.title("Distribution of articles by category")
sns.despine()


# plot the distribution of article token counts (using the raw unprocessed text)


# visualise the noun phrases



# load the english tokeniser, tagger, parser and NER
nlp = spacy.load("en_core_web_lg")
docs = [text for text in train.Text]
processed_docs = [nlp(text) for text in train.Text]
# extract noun chunks
noun_phrases = []
for doc in processed_docs:
    phrases = [chunk.text for chunk in doc.noun_chunks]
    noun_phrases.append(phrases)

# text is already in lower-case so no need to do anything
tokens_list = []
lemma_list = []
is_alphanum = []
is_punctuation = []
is_stopword = []
clean_docs = []
clean_docs2 = []
for doc in processed_docs:
    # split text into tokens based on white space (with a couple of in-built exceptions)
    tokens = [token.text for token in doc]
    tokens_list.append(tokens)
    # identify the lemmatised version of each token
    lemmas = [token.lemma_ for token in doc]
    lemma_list.append(lemmas)
    # is the token alphanumeric?
    alphanums = [token.is_alpha for token in doc]
    is_alphanum.append(alphanums)
    # is the token a punctuation symbol?
    puncs = [token.is_punct for token in doc]
    is_punctuation.append(puncs)
    # is the token a stopword
    stopwords = [token.is_stop for token in doc]
    is_stopword.append(stopwords)
    # retain only tokens in their lemmatised form that are alphanumeric, more than a single character, not punctuation
    # and not a stopword
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 1]
    clean_docs.append(clean_tokens)

from collections import Counter
word_counts = [Counter(doc).most_common(10) for doc in clean_docs]

https://www.machinelearningplus.com/spacy-tutorial-nlp/#textpreprocessingwithspacy
https://spacy.io/api/transformer
https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
https://www.machinelearningplus.com/nlp/custom-text-classification-spacy/
https://stackoverflow.com/questions/56281633/train-spacy-for-text-classification
    https: // www.width.ai / post / spacy - text - classification


# check if textcat is part of the pipeline
if nlp.has_pipe("textcat"):
    pass
else:
    textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
    textcat = nlp.add_pipe("textcat")
    print(nlp.pipe_names)
    textcat.add_label("business")
    textcat.add_label("sports")
    textcat.add_label("tech")
    textcat.add_label("politics")
    textcat.add_label("entertainment")

    # we can see that the distribution is fairly even, which is good, but business and sports articles are slightly more
    # common than tech, politics or entertainment articles

doc = nlp()

text_cats = [textcat(doc) for doc in processed_docs]

textcat(processed_docs[0])

# split on white spaces
# remove numbers, special characters, punctuation, single letter words
# remove stopwords
# stem (as we don't need the words to make sense)


#
# As we did not learn natural language processing (NLP) specific techniques such as word embeddings in the lectures,
# you will need to read discussions and example code from others in the Kaggle and/or other online research to make
# sure you understand. You can refer to any resource as needed, but make sure you “demonstrate” your understanding by
# explaining and interpreting in your own words. Also, include a reference list at the end of the report.

## Task 2: Build and train models
# In the Kaggle competition, the training data has labels. Thus, it can be solved using supervised learning. In
# general, the more labelled data we have, the more accurate the supervised learning model will be. But unsupervised
# learning can be powerful even when there is a small number of labels or no labels. This assignment will apply an
# unsupervised approach - the matrix factorization method - to discover topics in the news articles and use
# the labels to check the accuracy.
#
# Here are some steps to guide this section:
# 1) Think about this and answer: when you train the unsupervised model for matrix factorisation, should you include
# texts (word features) from the test dataset or not as the input matrix? Why or why not?
# 2) Build a model using the matrix factorisation method(s) and predict the train and test data labels. Choose any
# hyperparameter (e.g., number of word features) to begin with.
# 3) Measure the performances on predictions from both train and test datasets. You can use accuracy, confusion matrix,
# etc., to inspect the performance. You can get accuracy for the test data by submitting the result to Kaggle.
# 4) Change hyperparameter(s) and record the results. We recommend including a summary table and/or graphs.
# 5) Improve the model performance if you can - some ideas may include but are not limited to; using different feature
# extraction methods, fit models in different subsets of data, ensemble the model prediction results, etc.

## Task 3: Compare with the supervised learning approach
# 1) Pick and train a supervised learning method(s) and compare the results (train and test performance)
# 2) Discuss comparison with the unsupervised approach. You may try changing the train data size (e.g., Include only
# 10%, 20%, 50% of labels, and observe train/test performance changes). Which methods are data-efficient (require a
# smaller amount of data to achieve similar results)? What about overfitting?

## Task 4: Submit deliverables
# 1) A high quality Jupyter notebook or pdf report
# 2) A link to the github repo
# 3) A screenshot of the Kaggle challenge leaderboard



