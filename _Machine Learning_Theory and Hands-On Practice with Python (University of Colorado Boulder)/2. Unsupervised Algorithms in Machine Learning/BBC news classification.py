# get rid of the seaborn FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from wordcloud import WordCloud

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

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
# Here, we're going to follow a more classical approach to pre-processing the text.


def process_text(data_series, alphanumeric_tokens = True, single_tokens = False):
    # apply the spacy model to the input text data
    processed_docs = [nlp(text) for text in data_series]

    clean_docs = []
    clean_text = []
    for doc in processed_docs:
        if alphanumeric_tokens == True:
            if single_tokens == False:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha and len(token) > 1]
            else:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        else:
            if single_tokens == False:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and len(token) > 1]
            else:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        clean_docs.append(clean_tokens)

        clean_strings = ' '.join(clean_tokens)
        clean_text.append(clean_strings)

    return clean_docs, clean_text


# convert list to strings
clean_docs, clean_text = process_text(train.Text)
train['CleanTokens'] = clean_docs
train['CleanText'] = clean_text

# an example of the original text
doc_idx = 1489
train.Text[doc_idx]
# the same example but pre-processed
train.CleanText[doc_idx]
train.CleanTokens[doc_idx]


## Task 3: Feature Extraction
# As mentioned previously, computers don't understand text and need it to be converted into a numeric representation
# before it can be input into a machine learning algorithm. One of the simplest approach is 'Bag of Words', which
# represents each document as the count of each token in the entire dataset, where a token could be a single word
# and/or a longer n-gram. One slight variation on this idea is to weight tokens by how frequently they appear in a
# particular documents by how many documents in the datasets they are in. The idea behind the approach is two-fold:
# firstly, more important terms should receive a words that are less important. Secondly, a term should be considered
# more important if it appears frequently in a subset of the documents, and less important if it either appears
# infrequently or appears frequently across every document. The formal name for the approach is Term Frequency-Inverse
# Document Frequency or TF-IDF vectorisation, and is the text feature generation approach that will be used for the
# analysis.
#
# In addition, the EDA indicated that there were some very long articles in the dataset. These will be removed in case
# the algorithm mistakenly uses the length of the articles to infer something about how they should be categorised.

# remove articles in the top 1%
# this has very little impact on the mean and median, but brings the maximum word count down significantly (from
# 3,345 to 464)
top = np.quantile(train.NumWords, q=0.99)
train_sub = train.loc[train.NumWords < top]

# create a tfidf vectoriser with default arguments (extracts single word tokens, doesn't remove stopwords, expects
# strings, use all features etc)
train_vec = TfidfVectorizer()
train_mat = train_vec.fit_transform(train.CleanText)

## Task 4: Unsupervised Learning - Matrix Factorisation

# Yes, we need to include text / word features from the test set in the input matrix as there may be features in the
# test set only. If the model comes across new features it hasn't seen before during model training then it won't know
# how to handle them. This isn't classified as data leakage as we aren't predicting anything and hence not using
# ground truth labels to help the model learn.

# combine the train and test datasets together to build a Tf-Idf matrix with the entire vocabulary
train_test = pd.concat([train[['ArticleId', 'Category', 'Text']], test[['ArticleId', 'Text']]])
# process the raw strings
_, clean_text = process_text(train_test.Text)
train_test['CleanText'] = clean_text
# use the default parameters for now (we'll play with them later)
train_test_vec = TfidfVectorizer()
# apply the vectoriser to the combined dataset
train_test_mat = train_test_vec.fit_transform(train_test.CleanText)

# if we take a look at the shape of the resulting matrix, we can see that we have 21,970 tokens across 2,225 articles
train_test_mat.shape

# Now we're going to use sklearn's implementation of non-negative matrix factorisation (NMF) to extract the topic
# structure sklearn's implementation is chosen due to compatibility with the feature generator and the relative
# simplicity of using other sklearn functions like confusion matrices to assess model performance.
# Recall that the idea of NMF is to extract an additive model of the topic structure of the corpus. The output is a
# list of topics, with each topic represented by a list of terms (words in this case).
# The dimensionality of the problem and hence the runtime can be controlled by the n_samples, n_features and n_topics
# hyperparameters.

# we know that there are 5 topics, so we can explicitly set this
nmf_init = NMF(n_components=5, init='ssdsvda', max_iter=200, random_state=42)


https://scikit-learn.org/0.15/auto_examples/applications/topics_extraction_with_nmf.html
https://sandipanweb.wordpress.com/2023/12/06/non-negative-matrix-factorization-to-solve-text-classification-and-recommendation-problems/




Hello there!👋🏻

What’s up, fellow language enthusiasts? Today we’re gonna talk about something that’s super important for anyone dealing with large volumes of text: topic modeling and text classification. It’s all about making sense of the information overload, you know what I mean?

But here’s the thing: doing this manually is a real pain in the neck. That’s where Non-Negative Matrix Factorization (NMF) comes in. It’s a mathematical technique that can help us extract meaningful topics from a bunch of documents and classify them automatically.

And the best part is, NMF doesn’t just save us a ton of time — it can also improve the accuracy of our results. So if you’re tired of slogging through endless piles of text, stick around and let’s dive into how NMF can make your life easier.🚀
Photo by Egor Myznik on Unsplash
The Theory Behind NMF

Okay, so let’s get a bit more technical now. Matrix factorization is a technique used in linear algebra to decompose a matrix into a product of two or more matrices. The goal is to simplify the original matrix and extract its underlying structure.

NMF is a specific type of matrix factorization that is particularly useful for non-negative data. It works by decomposing a non-negative matrix into two non-negative matrices: a “basis” matrix and a “weights” matrix. The basis matrix represents the underlying topics or patterns in the data, while the weights matrix represents how strongly each document is associated with those topics.

The mathematical equations involved in NMF can be a bit intimidating, but the basic idea is that we start with a matrix of word frequencies in a set of documents, and then iteratively adjust the basis and weights matrices to minimize the difference between the original matrix and their product. This process converges on a solution that represents the most important topics in the data.
Photo by Andreas Fickl on Unsplash
Applications of NMF in Topic Modeling and Text Classification

Now that we understand how NMF works, let’s talk about how it’s used in topic modeling and text classification. The basic idea is to represent each document as a linear combination of topics, where the topics are represented by the columns of the basis matrix.

For example, let’s say we have a set of news articles about politics, sports, and entertainment. We can use NMF to extract the most important topics from these articles, and then classify each article based on which topics it is most strongly associated with. This allows us to automatically categorize large volumes of text without having to read every single document.

NMF can also be used for more advanced applications, such as clustering similar documents together, identifying the most important keywords for each topic, and even generating new text based on existing patterns in the data.
Photo by ThisisEngineering RAEng on Unsplash
How to Implement NMF

Implementing NMF can be a bit tricky, especially if you’re not familiar with linear algebra or machine learning. However, there are many software packages and libraries available that can make it easier. Some popular options include scikit-learn and TensorFlow in Python and the NMF package in R.

The basic steps involved in implementing NMF are:

    Preprocess the text data to remove stop words, stem or lemmatize words, and convert the text to a numerical format (e.g. using TF-IDF or bag-of-words).
    Choose the number of topics you want to extract and initialize the basis and weights matrices.
    Iteratively update the basis and weights matrices using a cost function that measures the difference between the original matrix and their product.
    Evaluate the resulting topics and use them to classify new text data.

Case Studies

There have been many case studies that demonstrate the effectiveness of NMF in topic modeling and text classification. For example, one study used NMF to extract topics from a large set of scientific articles and found that it was able to identify important themes more accurately than other methods. Another study used NMF to classify customer reviews of products and found that it outperformed other techniques in terms of accuracy.
Advantages and Limitations of NMF

The advantages of NMF in topic modeling and text classification are clear: it can save a lot of time and improve the accuracy of results. However, there are also some limitations to consider. For example, NMF can be sensitive to the choice of parameters and initialization, and it may not work as well with very sparse or noisy data.

To address these limitations, it’s important to carefully choose the number of topics and the preprocessing steps used and to experiment with different initializations and cost functions.
Conclusion

Overall, Non-Negative Matrix Factorization (NMF) is a powerful technique for topic modeling and text classification that can save time and improve the accuracy of results. By representing text data as a matrix and iteratively decomposing it into basis and weights matrices, NMF can extract meaningful topics from large volumes of text and classify documents based on those topics.

Implementing NMF can be challenging, but there are many software packages and libraries available to help. It’s also important to carefully choose the number of topics and preprocessing steps and to experiment with different initializations and cost functions.

NMF has been successfully applied in many case studies, including scientific article analysis and product review classification. While there are limitations to consider, such as sensitivity to parameters and sparse or noisy data, NMF remains a valuable tool for anyone dealing with large volumes of text. So next time you’re faced with a mountain of documents to classify, consider giving NMF a try!

Thanks to all who have read, follow me for interesting articles about machine learning👋🏻😊





The default parameters (n_samples / n_features / n_topics) should make the example runnable in a couple of tens of seconds. You can try to increase the dimensions of the problem, but be aware than the time complexity is polynomial.


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



