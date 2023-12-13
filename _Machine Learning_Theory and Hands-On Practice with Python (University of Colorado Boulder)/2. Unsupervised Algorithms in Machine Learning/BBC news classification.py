# get rid of the seaborn FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from wordcloud import WordCloud

# check the backend and change if required
import matplotlib as mpl

mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass


## Task 1: Exploratory Data Analysis
# load the data
train = pd.read_csv('data/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('data/learn-ai-bbc/BBC News Test.csv')

# take a look at the first few rows of each dataset - notice that the test dataset doesn't include labels so we're not
# going to be able to use it for model testing. Instead, we're going to have to rely on the training dataset.
# also note that all the text is already in lower-case, which means it doesn't have to be included in the data
# # cleaning.
train.head()
test.head()

# check out the distribution of category labels
# we can see that the distribution is fairly even, which is good, but business and sports articles are slightly more
# common than tech, politics or entertainment articles
sns.countplot(data=train, x="Category", color="seagreen")
plt.title("Distribution of articles by category")
sns.despine()

# check out the number of tokens in the articles
train['NumChars'] = train.Text.apply(lambda x: len(x))
train['NumWords'] = train.Text.apply(lambda x: len(x.split()))

# tech and politics tend to have the largest number of words per article
# however, the distributions by category are fairly similar i.e. there are more shorter articles and a relatively long
# right tail to the distribution with a couple of significant outliers
train.groupby('Category')['NumWords'].median()

fig, axs = plt.subplots(1, 2)
sns.boxplot(data=train, x="Category", y="NumWords", showfliers=False, color="seagreen", ax=axs[0])
plt.title("Distribution of word counts by category")
sns.despine()

sns.histplot(data=train, x="NumWords", hue="Category", color="seagreen", multiple="stack", palette="Greens", ax=axs[1])
plt.title("Distribution of word counts by category")
sns.despine()

# load the english spacy model
nlp = spacy.load("en_core_web_lg")
# apply the spacy model to the training data
processed_docs = [nlp(text) for text in train.Text]

# extract noun chunks
doc_noun_phrases = []
doc_noun_phrases_joined = []
for doc in processed_docs:
    phrases = [chunk.text for chunk in doc.noun_chunks]
    phrases2 = [phrase.replace(" ", "_") for phrase in phrases]
    doc_noun_phrases.append(phrases)
    doc_noun_phrases_joined.append(phrases2)
# noun_phrases = [phrases for doc in doc_noun_phrases for phrases in doc]
# noun_phrases_joined = [phrases for doc in doc_noun_phrases_joined for phrases in doc]
train['NounPhrases'] = doc_noun_phrases
train['NounPhrasesJoined'] = doc_noun_phrases_joined

# create a version without single words
doc_noun_phrases = []
doc_noun_phrases_joined = []
for doc in train.NounPhrases:
    # lengths = [len(token.split()) for token in doc]
    phrases = [token for token in doc if len(token.split()) > 1]
    phrases2 = [phrase.replace(" ", "_") for phrase in phrases]
    doc_noun_phrases.append(phrases)
    doc_noun_phrases_joined.append(phrases2)
train['NounPhrasesSub'] = doc_noun_phrases
train['NounPhrasesJoinedSub'] = doc_noun_phrases_joined

# convert list to strings
train['NounStrings'] = train.NounPhrasesJoined.apply(lambda x: " ".join(x))
train['NounStringsSub'] = train.NounPhrasesJoinedSub.apply(lambda x: " ".join(x))

# create a colour map dictionary and convert the RGB tuples to hexidecimal
color_dict = {'business': sns.color_palette("crest")[0],
              'entertainment': sns.color_palette("crest")[1],
              'politics': sns.color_palette("crest")[2],
              'sport': sns.color_palette("crest")[3],
              'tech': sns.color_palette("crest")[4]}


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

color_dict = {category: rgb_to_hex(rgb) for category, rgb in color_dict.items()}

# group and join the phrases from each category
grouped_train_sub = train.groupby('Category')['NounStringsSub'].apply(lambda x: ' '.join(x)).reset_index()
grouped_train = train.groupby('Category')['NounStrings'].apply(lambda x: ' '.join(x)).reset_index()

# plot a grid of word clouds
# even without the headings, it would be reasonably clear what the different categories were about: 'business' mentions
# things like markets, companies, the economy; 'entertainment' mentions films, shows and awards; 'politics' is very
# UK-specific with things the tories and Tony Blair; 'sport' mentions games, matches and players; and 'tech'
# mentions things like technology, service, internet and names of countries
fig, axs = plt.subplots(3, 2, figsize=(15,10))
for idx, (ax, (category, row)) in enumerate(zip(axs.flatten(), grouped_train_sub.iterrows())):
    category = row['Category']
    phrases = row['NounStringsSub']

    wc = WordCloud(color_func=lambda *args, **kwargs: color_dict[category])
    wc.generate(phrases)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'{category}', fontsize=12)
fig.suptitle('Noun Phrase Word Clouds by Category')
# remove the last empty plot
fig.delaxes(axs.flatten()[-1])
plt.tight_layout()


fig, axs = plt.subplots(3, 2, figsize=(15,10))
for idx, (ax, (category, row)) in enumerate(zip(axs.flatten(), grouped_train.iterrows())):
    category = row['Category']
    phrases = row['NounStrings']

    wc = WordCloud(color_func=lambda *args, **kwargs: color_dict[category])
    wc.generate(phrases)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f'{category}', fontsize=12)
fig.suptitle('Noun Phrase Word Clouds by Category')
# remove the last empty plot
fig.delaxes(axs.flatten()[-1])
plt.tight_layout()


## Task 2: Data Pre-Processing
# The pre-processing for text data is typically slightly different to how it would be approached for a regular tabular
# dataset. Specifically, it needs to be essentially converted into a numeric representation that a machine can
# understand. Techniques employed to do this include things like converting all text into a single case (usually
# lower-case), splitting text into individual words (tokenising), removing extraneous information like punctuation,
# numbers and symbols, and 'noise' words that provide next-to-no extra information (stopwords). However, more modern
# approaches using Large Language Models for example, often use this information to provide extra context, so whilst
# they may still tokenise the text, the removal of certain tokens may not occur. In addition, there is other
# processing like padding out each tokenised string to the same length and adding a mask to differentiate between the
# content and the padding.
#
# Here, we're going to follow a more classical approach to pre-processing the text. In addition, the EDA indicated that
# there were some very long articles - we'll remove them to help (in theory) make the classifier's task a little
# easier.

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
    # retain only tokens in their lemmatised form that aren't necessarily alphanumeric but follow the other
    # restrictions
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token) > 1]
    clean_docs2.append(clean_tokens)
# convert list to strings
train['CleanDocsAlphanum'] = clean_docs
train['CleanDocs'] = clean_docs2

# remove articles in the top 1%
# this has very little impact on the mean and median, but brings the maximum word count down significantly (from
# 3,345 to 464)
top = np.quantile(train.NumWords, q=0.99)
train_sub = train.loc[train.NumWords < top]


## Task 3: Generate Word Embeddings


## Task 4: Unsupervised Learning - Matrix Factorisation

# Yes, we need to include text / word features from the test set in the input matrix as there may be features in the
# test set only. If the model comes across new features it hasn't seen before during model training then it won't know
# how to handle them. This isn't classified as data leakage as we aren't predicting anything and hence not using
# ground truth labels to help the model learn.


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



