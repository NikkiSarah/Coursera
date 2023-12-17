# get rid of the seaborn FutureWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# general libraries
import configparser
from itertools import product
import numpy as np
import pandas as pd
from time import time

# visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# modelling libraries
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin

# check the backend and change if required
import matplotlib as mpl
mpl_backend = mpl.get_backend()
if mpl_backend != "Qt5Agg":
    mpl.use("Qt5Agg")
else:
    pass


## Task 1: Exploratory Data Analysis
# load the data
train = pd.read_csv('./inputs/learn-ai-bbc/BBC News Train.csv')
test = pd.read_csv('./inputs/learn-ai-bbc/BBC News Test.csv')

# take a look at the first few rows of each dataset - notice that the test dataset doesn't include labels so we're not
# going to be able to use it for model testing. Instead, we're going to have to rely on the training dataset.
# also note that all the text is already in lower-case, which means it doesn't have to be included in the data
# cleaning.
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
t0 = time()
# apply the spacy model to the training data
processed_docs = [nlp(text) for text in train.Text]
print("done in %0.3fs." % (time() - t0))

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
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and
                                token.is_alpha and len(token) > 1]
            else:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and
                                token.is_alpha]
        else:
            if single_tokens == False:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and
                                len(token) > 1]
            else:
                clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        clean_docs.append(clean_tokens)

        clean_strings = ' '.join(clean_tokens)
        clean_text.append(clean_strings)

    return clean_docs, clean_text


# convert list to strings
t0 = time()
clean_docs, clean_text = process_text(train.Text)
train['CleanTokens'] = clean_docs
train['CleanText'] = clean_text
print("done in %0.3fs." % (time() - t0))

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
train_mat = train_vec.fit_transform(train_sub.CleanText)

## Task 4: Unsupervised Learning - Matrix Factorisation
# We should include text / word features from the test set in the input matrix as there may be features in the
# test set only. If the model comes across new features it hasn't seen before during model training then it won't know
# how to handle them. This isn't classified as data leakage as we aren't predicting anything and hence not using
# ground truth labels to help the model learn.

# combine the train and test datasets together to build a Tf-Idf matrix with the entire vocabulary
train_test = pd.concat([train[['ArticleId', 'Category', 'Text']], test[['ArticleId', 'Text']]])
# count the number of words in each article and remove the top 1%
train_test['NumWords'] = train_test.Text.apply(lambda x: len(x.split()))
train_test.NumWords.describe()
train_test_sub = train_test.loc[train_test.NumWords < top]
# again has very little impact on the mean/median, but removes all the very long outlying articles
train_test_sub.NumWords.describe()

# process the raw strings
t0 = time()
_, clean_text = process_text(train_test_sub.Text)
train_test_sub.loc[:, 'CleanText'] = clean_text
print("done in %0.3fs." % (time() - t0))

# use the default parameters for now (we'll play with them later)
tfidf_vec = TfidfVectorizer()
# apply the vectoriser to the combined dataset
train_test_mat = tfidf_vec.fit_transform(train_test_sub.CleanText)
# get the vocab (this is important for identifying the top terms in a topic)
vocab = np.array(tfidf_vec.get_feature_names_out())
# vocab = tfidf_vec.vocabulary_  # this is a dictionary representation

# We can see that we have 21,389 tokens across 2,202 articles. If we wanted to reduce the number of terms in the
# vocabulary, we could filter out terms that only appear in 1 article and/or 95% of articles for example.
# Note that the number of features in the matrix is the same as the number of terms in the vocabulary.
train_test_mat.shape

# Now we're going to use sklearn's implementation of non-negative matrix factorisation (NMF) to extract the topic
# structure. NMF decomposes a document feature matrix into two component matrices that are iteratively adjusted until
# the difference between the original matrix and the product of these component matrices is minimised. What this means
# for text classification is that each document is represented as a linear combination of topics, and then each
# document is given the label of the most representative topic. Other matrix factorisation methods include latent
# semantic analysis (TruncatedSVD in sklearn) or Latent Dirichlet Allocation. Whilst LDA is a popular algorithm, NMF
# can sometimes produce more coherent topics. The choice of algorithm for this analysis was largely random and the
# author wanted to try something other than the popular LDA and LSA approaches.
#
# Recall that the idea of NMF is to extract an additive model of the topic structure of the corpus. The output is a
# list of topics, with each topic represented by a list of terms (words in this case).
# The dimensionality of the problem and hence the runtime can be controlled by the number of documents, number of
# topics (n_components) and the number of features in the vectoriser (max_features).

# we know that there are 5 topics, so we can explicitly specify this
nmf_init = NMF(n_components=5, init='nndsvd', random_state=42)
# extract the component matrices
nmf_W1_document_topics = nmf_init.fit_transform(train_test_mat)
nmf_H1_topic_terms = nmf_init.components_

# take a look at the top terms from each topic and map them to the known categories
num_terms = 20
topic_terms = pd.DataFrame(np.apply_along_axis(lambda x: vocab[(np.argsort(-x))[:num_terms]], 1, nmf_H1_topic_terms))
topic_terms.reset_index(inplace=True)
topic_terms.rename(columns={'index': 'Category'}, inplace=True)
topic_dict = {0: 'sport', 1: 'politics', 2: 'tech', 3: 'entertainment', 4: 'business'}
topic_terms['Category'] = topic_terms.Category.map(topic_dict)


# Use the model to predict the labels of the train and test set
def generate_predictions(vectoriser, df, text_col, model, label_col, topic_dict):
    vectors = np.array(vectoriser.transform(df[text_col]).todense())
    preds = model.transform(vectors)
    pred_df = pd.DataFrame(np.argmax(preds, axis=1).reshape(-1, 1), columns=['Predicted'])
    pred_df['PredictedLabel'] = pred_df.Predicted.map(topic_dict)
    pred_df['ArticleId'] = df['ArticleId']

    if label_col is None:
        pass
    else:
        topic_dict_rev = {'sport': 0, 'politics': 1, 'tech': 2, 'entertainment': 3, 'business': 4}
        pred_df['Actual'] = df[label_col].map(topic_dict_rev)
        pred_df['ActualLabel'] = pred_df.Actual.map(topic_dict)

    return pred_df


train_preds = generate_predictions(tfidf_vec, train, 'CleanText', nmf_init, 'Category', topic_dict)

# Calculate and display a confusion matrix and some accuracy metrics like F1 and accuracy
def plot_confusion_matrix(true_labels, predicted_labels, topic_dictionary):
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    cm_df = pd.DataFrame(cm, columns=topic_dictionary, index=topic_dictionary)
    # cm_df = pd.DataFrame(cm, columns=topic_dictionary, index=topic_dictionary)
    ax = sns.heatmap(cm_df, annot=True, fmt=".0f", cbar=False, cmap="Greens")
    ax.set(xlabel="True Label", ylabel="Predicted Label",
           title='Non-Negative Matrix Factorisation Training Data Confusion Matrix')

# It looks like the algorithm does pretty well. Accuracy is around 92% on the training data. The problem areas are
# mistaking business articles for politics and tech articles, and a little bit surprisingly, entertainment articles for
# tech articles. When submitted to the kaggle platform, accuracy increased slightly to around 93%, confirming that the
# model was already doing reasonably well out-of-the-box.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, list(topic_dict.values()))

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
accuracy

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
f1

# preprocess the test data
t0 = time()
_, clean_text = process_text(test.Text)
test.loc[:, 'CleanText'] = clean_text
print("done in %0.3fs." % (time() - t0))

test_preds = generate_predictions(tfidf_vec, test, 'CleanText', nmf_init, None, topic_dict)
kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission nmf init.csv', index=False)


## Task 5: Hyperparameter Tuning
# However, can we do even better? We'll use a grid search approach to test whether changing some of the hyperparameters
# improves performance. The hyperparameters will include the initialisation method, the solver, the beta divergence
# method, W and H regularisation, and regularisation mixing.

param_grid = {'init': ('nndsvd', 'nndsvda', 'nndsvdar', 'random'),
              'solver': ('mu', 'cd'),
              'beta_loss': ('kullback-leibler', 'frobenius'),
              'alpha_W': [0, 0.0001, 0.001, 0.01, 0.1, 1],
              'alpha_H': ['same', 0, 0.0001, 0.001, 0.01, 0.1, 1],
              'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

t0 = time()

param_list = []
acc_list = []
f1_list = []

for init, solver, loss, aW, aH, l1 in product(param_grid['init'], param_grid['solver'], param_grid['beta_loss'],
                                              param_grid['alpha_W'], param_grid['alpha_H'], param_grid['l1_ratio']):
    # combinations that I know don't work
    if (solver == 'mu' and init == 'nndsvd') or (solver == 'cd' and loss == 'kullback-leibler') or \
        (solver == 'cd' and init in ['nndsvd', 'nndsvda'] and loss == 'frobenius' and aW > 0 and l1 > 0) or \
        (solver == 'cd' and init in ['nndsvd', 'nndsvda', 'nndsvdar'] and loss == 'frobenius' and aW == 0 and
            (aH in [0.01, 0.1, 1]) and l1 > 0) or \
        (solver == 'mu' and init == 'nndsvda' and loss == 'frobenius' and aW > 0 and aH != 0 and l1 > 0) or \
        (init == 'nndsvdar' and loss == 'frobenius' and aW > 0 and aH != 0 and l1 > 0) or \
        (init == 'random'):
        pass
    else:
        params = [init, solver, loss, aW, aH, l1]
        nmf_tuned = NMF(n_components=5, init=init, solver=solver, beta_loss=loss, max_iter=1000, random_state=42,
                        alpha_W=aW, alpha_H=aH, l1_ratio=l1)
        nmf_tuned.fit(train_test_mat)
        try:
            # predict the training data labels
            train_preds = generate_predictions(tfidf_vec, train, 'CleanText', nmf_tuned, 'Category', topic_dict)
            # calculate the accuracy
            train_acc = accuracy_score(train_preds['ActualLabel'], train_preds['PredictedLabel'])
            train_f1 = f1_score(train_preds['ActualLabel'], train_preds['PredictedLabel'], average='weighted')

            # print and save the results of the better combinations
            if train_acc > 0.5:
                print(params, round(train_acc, 4))

                param_list.append(params)
                acc_list.append(train_acc)
                f1_list.append(train_f1)
        # ValueError: Array passed to NMF (input H) is full of zeros.
        except ValueError:
            print(params, "Error raised")

print("done in %0.3fh." % ((time() - t0)/60/60))
# last completed in 1.58 hours

results = pd.DataFrame(data={'hyperparameters': param_list, 'accuracy_score': acc_list,
                             'f1_score': f1_list})
results.sort_values(by='accuracy_score', inplace=True, ascending=False)
results.to_csv("./outputs/nmf tuning results.csv", index=False)

# select the best combination and retrain the model to predict the labels on the train and test set
# it's nice to see there's no conflict between the best model based on accuracy vs F1
best_params = results.head(1)
best_params_f1 = results.sort_values(by='f1_score', ascending=False).head(1)

nmf_final = NMF(n_components=5, init='nndsvda', solver='mu', beta_loss='kullback-leibler', max_iter=500,
                random_state=42, alpha_W=1, alpha_H=0.1, l1_ratio=0)
nmf_final.fit(train_test_mat)

train_preds = generate_predictions(tfidf_vec, train, 'CleanText', nmf_final, 'Category', topic_dict)
test_preds = generate_predictions(tfidf_vec, test, 'CleanText', nmf_final, None, topic_dict)

kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission nmf final.csv', index=False)

# It looks as though the tuning has worked and model accuracy has improved to around 94% on the training data. The
# algorithm is more successful at correctly classifying entertainment articles (recall the initial model liked to
# classify them as tech articles), but is still missclassifying business articles as either politics or tech articles.
# When submitted to the kaggle platform, accuracy increased marginally but was still approximately 93%.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, list(topic_dict.values()))
cm = confusion_matrix(train_preds.Actual, train_preds.Predicted)
cm

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
accuracy

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
f1


## Task 6: Supervised Learning - Spacy
# Spacy has an inbuilt text categoriser ('textcat') that can be added as a component to its NLP pipeline.

# There are 3 CPU architectures available:
# - Stacked ensemble of a linear BoW and neural network model. The neural network is built on top of a Tok2Vec (token
#   to vector) layer and uses attention. This is the default architecture.
# - Neural network model where token vectors are calculated using a CNN. According to the documentation, it's typically
#   less accurate than the default but faster.
# - n-gram BoW model. Runs the fastest, but has particular trouble with short texts (Not too much of an issue in this
#   case, but could be if analysing customer feedback for example).

# If a GPU is available, then a transformer model from the HuggingFace transformers library with pre-trained weights
# and a PyTorch implementation can be added.

# The first model to be trained and evaluated will be a BoW model without a GPU.

# check if textcat is part of the pipeline
if nlp.has_pipe("textcat"):
    pass
else:
    textcat = nlp.add_pipe("textcat", last=True)
print(nlp.pipe_names)
# add the labels (categories) to the pipeline component
textcat.add_label("business")
textcat.add_label("entertainment")
textcat.add_label("politics")
textcat.add_label("sport")
textcat.add_label("tech")

# double-check they've been added
textcat.labels

default_config = nlp.get_pipe_meta("textcat").default_config

# rejig the config file for efficiency, note that changes the default model architecture from an ensemble to a BoW.
# Step 1: generate a base config file using https://spacy.io/usage/training#quickstart:
# Components: textcat
# Text classification: Exclusive categories
# Hardware: CPU
# Optimise for: efficiency
# Step 2: Create a complete config file with all the other components auto-filled to their defaults
# CLI command python -m spacy init fill-config ./inputs/base_efficiency_cpu_config.cfg ./inputs/efficiency_cpu_config.cfg --diff
# --diff produces a helpful comparison to the base config file (i.e. what's been added/removed)

# view the configuration file
def read_spacy_config(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for option in config.options(section):
            config_dict[section][option] = config.get(section, option)

    return config_dict

file_path = './inputs/efficiency_cpu_config.cfg'
textcat_config = read_spacy_config(file_path)

# split the data into a training and dev set (80/20 split)
# ensure that the category splits are roughly even
train_data, dev_data = train_test_split(train_sub, stratify=train_sub.Category, test_size=0.2,
                                        random_state=42)
train_data.reset_index(inplace=True, drop=True)
dev_data.reset_index(inplace=True, drop=True)

# convert the data to spacy's required training format: https://spacy.io/api/data-formats#binary-training
def convert_data(output_path, df, label='Category'):
    # extract all the unique categories
    cats = set(df[label])

    # create a one-hot-dictionary for each unique category
    one_hot_dicts = {}
    for c1 in cats:
        one_hot_dict = {c2: (1 if c2 == c1 else 0) for c2 in cats}
        one_hot_dicts[c1] = one_hot_dict
    print(one_hot_dicts)

    # create spacy and DocBin objects
    nlp = spacy.blank('en')
    db = DocBin()

    # for each row in the dataframe...
    for idx, row in df.iterrows():
        # locate just the text and label information
        text = row['Text']
        cat = row['Category']

        # make a doc from the text
        doc = nlp.make_doc(text)
        # add the relevant one-hot-dictionary
        doc.cats = one_hot_dicts[cat]
        # print(one_hot_dicts[cat])
        # add it to the DocBin object
        db.add(doc)

    # write the DocBin object to disk
    db.to_disk(output_path)

convert_data("./inputs/train.spacy", df=train_data)
convert_data("./inputs/dev.spacy", df=dev_data)

# train the model: https://spacy.io/usage/training#quickstart
# override various sections using the syntax: --section.option
# python -m spacy train ./inputs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./inputs/train.spacy --paths.dev ./inputs/dev.spacy

# ============================= Training pipeline =============================
# ℹ Pipeline: ['textcat']
# ℹ Initial learn rate: 0.001
# E    #       LOSS TEXTCAT  CATS_SCORE  SCORE
# ---  ------  ------------  ----------  ------
#   0       0          0.16        5.90    0.06
#   0     200         15.19       90.06    0.90
#   0     400          4.26       91.09    0.91
#   0     600          2.46       95.02    0.95
#   0     800          2.45       92.84    0.93
#   0    1000          2.98       95.78    0.96
#   1    1200          1.62       96.89    0.97
#   1    1400          0.39       95.66    0.96
#   1    1600          0.45       96.18    0.96
#   1    1800          0.00       96.83    0.97
#   1    2000          1.15       95.25    0.95
#   1    2200          0.22       96.91    0.97
#   2    2400          0.00       97.64    0.98
#   2    2600          0.60       96.57    0.97
#   2    2800          0.00       97.64    0.98
#   2    3200          0.11       95.87    0.96
#   2    3400          0.00       96.97    0.97
#   3    3600          0.00       97.99    0.98
#   3    3800          0.00       97.99    0.98
#   3    4000          0.00       97.65    0.98
#   3    4200          0.00       97.99    0.98
#   3    4400          0.40       97.99    0.98
#   3    4600          0.00       97.65    0.98
#   4    4800          0.00       97.99    0.98
#   4    5000          0.00       97.65    0.98
#   4    5200          0.40       97.65    0.98

# evaluate model performance
# python -m spacy evaluate ./outputs/efficiency-cpu-model-best/ ./inputs/train.spacy
# python -m spacy evaluate ./outputs/efficiency-cpu-model-best/ ./inputs/dev.spacy

# According to these commands, the model had the following on the training data:
# TOK                 100.00
# TEXTCAT (macro F)   99.92
# SPEED               239491

# And the following on the dev data:
# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   97.99
# SPEED               168510
# We can see therefore that the scoring during model training is for the dev set (as it should be). Also, the F-score
# on the dev set is only slightly lower than the training set, indicating that it's not overfitting.

nlp_efficiency = spacy.load("./outputs/efficiency-cpu-model-best")

def generate_predictions(df, text_col='Text', label_col='Category', textcat_model=nlp_efficiency):
    data = df[text_col].to_list()

    pred_cats = []
    for article in data:
        doc = textcat_model(article)
        scores_dict = doc.cats
        pred_cat = max(scores_dict, key=lambda k: scores_dict[k])
        pred_cats.append(pred_cat)
    if label_col is None:
        pred_df = pd.DataFrame(data = {'ArticleId': df['ArticleId'], 'PredictedLabel': pred_cats, 'Text': df[text_col]})
    else:
        pred_df = pd.DataFrame(data = {'ArticleId': df['ArticleId'], 'Actual': df[label_col],
                                       'Predicted': pred_cats, 'Text': df[text_col]})

    return pred_df

train_preds = generate_predictions(train)

def plot_confusion_matrix(true_labels, predicted_labels, title,
                          cats=['business', 'entertainment', 'politics', 'sport', 'tech']):
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    cm_df = pd.DataFrame(cm, columns=cats, index=cats)
    ax = sns.heatmap(cm_df, annot=True, fmt=".0f", cbar=False, cmap="Greens")
    ax.set(xlabel="True Label", ylabel="Predicted Label", title=title)

# It looks like the algorithm does very well. Accuracy is near 100% on the training data.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, title='Spacy n-gram Bag of Words Confusion Matrix')

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
accuracy

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
f1

# make predictions
test_preds = generate_predictions(test, label_col=None)

kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission efficient textcat.csv', index=False)

# The model performed very well on the test set, achieving an accuracy of around 97%.

# The second model to be trained and evaluated will be a stacked ensemble model without a GPU.
# rejig the config file for accuracy, note that keeps the default model architecture.
# Step 1: generate a base config file using https://spacy.io/usage/training#quickstart:
# Components: textcat
# Text classification: Exclusive categories
# Hardware: CPU
# Optimise for: accuracy
# Step 2: Create a complete config file with all the other components auto-filled to their defaults
# CLI command python -m spacy init fill-config ./inputs/base_accuracy_cpu_config.cfg ./inputs/accuracy_cpu_config.cfg --diff
# --diff produces a helpful comparison to the base config file (i.e. what's been added/removed)

# view the configuration file
file_path = './inputs/accuracy_cpu_config.cfg'
textcat_config = read_spacy_config(file_path)

# train the model: https://spacy.io/usage/training#quickstart
# override various sections using the syntax: --section.option
# python -m spacy train ./inputs/accuracy_cpu_config.cfg --output ./outputs --paths.train ./inputs/train.spacy --paths.dev ./inputs/dev.spacy

# ============================= Training pipeline =============================
# ℹ Pipeline: ['tok2vec', 'textcat']
# ℹ Initial learn rate: 0.001
# E    #       LOSS TOK2VEC  LOSS TEXTCAT  CATS_SCORE  SCORE
# ---  ------  ------------  ------------  ----------  ------
#   0       0          0.00          0.16        6.49    0.06
#   0     200         23.94         38.13       21.74    0.22
#   0     400         35.73         37.83       38.15    0.38
#   0     600         77.38         33.28       41.03    0.41
#   0     800         41.17         34.24       30.97    0.31
#   0    1000         42.21         35.12       38.06    0.38
#   1    1200         53.72         29.06       47.15    0.47
#   1    1400         29.86         28.71       69.74    0.70
#   1    1600         42.00         18.32       39.36    0.39
#   1    1800         39.36         20.43       81.93    0.82
#   1    2000         29.13         18.70       64.02    0.64
#   1    2200         68.32         22.09       83.33    0.83
#   2    2400         23.12         13.78       77.05    0.77
#   2    2600         50.87         15.56       78.93    0.79
#   2    2800         30.66         15.77       78.24    0.78
#   2    3000         70.07         12.74       83.93    0.84
#   2    3200         37.61         13.91       85.94    0.86
#   2    3400         24.35          7.46       83.30    0.83
#   3    3600         42.41          9.66       89.59    0.90
#   3    3800         23.04          7.31       81.86    0.82
#   3    4000         24.68          8.41       88.52    0.89
#   3    4200         35.47         12.20       91.80    0.92
#   3    4400         27.44          9.86       90.98    0.91
#   3    4600         15.84          4.01       89.92    0.90
#   4    4800         25.87          7.05       81.14    0.81
#   4    5000         49.66          6.63       85.44    0.85
#   4    5400         37.67          8.26       88.33    0.88
#   4    5600         26.57          4.37       92.77    0.93
#   4    5800         44.53          4.74       93.34    0.93
#   5    6000         31.11          4.95       90.12    0.90
#   5    6200         28.76          3.69       92.93    0.93
#   5    6400         44.18          3.94       86.14    0.86
#   5    6600         81.44          8.64       89.62    0.90
#   5    6800         85.67          6.02       90.78    0.91
#   5    7000         30.05          3.60       87.15    0.87
#   6    7200         37.27          5.41       90.87    0.91
#   6    7400         33.39          4.21       93.13    0.93

# evaluate model performance
# python -m spacy evaluate ./outputs/accuracy-cpu-model-best/ ./inputs/train.spacy
# python -m spacy evaluate ./outputs/accuracy-cpu-model-best/ ./inputs/dev.spacy

# According to these commands, the model had the following on the training data:
# TOK                 100.00
# TEXTCAT (macro F)   99.92
# SPEED               239491

# ================================== Results ==================================
#
# TOK                 100.00
# TEXTCAT (macro F)   93.34
# SPEED               4866

# We can see therefore that the scoring during model training is for the dev set (as it should be). Also, the F-score
# on the dev set is only slightly lower than the training set, indicating that it's not overfitting.

nlp_accuracy = spacy.load("./outputs/accuracy-cpu-model-best")

train_preds = generate_predictions(train, textcat_model=nlp_accuracy)

# It looks like the algorithm does reasonably well. Accuracy is about 95% on the training data. The main area of
# concern is that the classifier mistakes business articles for political articles.
plot_confusion_matrix(train_preds.Actual, train_preds.Predicted, title='Spacy Stacked Ensemble Confusion Matrix')

accuracy = accuracy_score(train_preds.Actual, train_preds.Predicted)
accuracy

f1 = f1_score(train_preds.Actual, train_preds.Predicted, average='weighted')
f1

# make predictions
test_preds = generate_predictions(test, label_col=None, textcat_model=nlp_accuracy)

kaggle_submission = test_preds.loc[:, ['ArticleId', 'PredictedLabel']]
kaggle_submission.columns=['ArticleId', 'Category']
kaggle_submission.to_csv('./outputs/kaggle submission accurate textcat.csv', index=False)

# The model performed poorly on the test set, achieving an accuracy of around 91%.

# Final comparison
Model: NMF
Efficiency: Inefficient (lots of pre-processing and hp tuning time-intensive)
Training speed: Fast
Train accuracy: 93.6%
Test accuracy: 92.3%
Overfitting: Unable to double-check as a train-dev split wasn't used during hp training, but a comparison of the
             train-test scores indicates it's not really present

Model: Spacy BoW
Efficiency: Reasonably efficient (a lot of the pre-processing occurs under the hood by calling nlp(doc) for example)
Training speed: Reasonably fast
Train accuracy: 99.5%
Test accuracy: 97.4%
Overfitting: No (99.9% on the training data and 98.0% on the dev data)

Model: Spacy Stacked Ensemble
Efficiency: As per Spacy BoW
Training speed: Much slower
Train accuracy: 95.3%
Test accuracy: 90.9%
Overfitting: No (xx.x% on the training data and 93.3% on the dev data)

# finally, observe the effect on performance for a BoW model when random subsets of the data are used
for ts in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    train_data, dev_data = train_test_split(train_sub, stratify=train_sub.Category, test_size=ts)
    train_data.reset_index(inplace=True, drop=True)
    dev_data.reset_index(inplace=True, drop=True)

    # convert the data to spacy's required training format: https://spacy.io/api/data-formats#binary-training
    train_path = "./inputs/train_experiment" + enumerate(ts) + ".spacy"
    dev_path = "./inputs/dev_experiment" + enumerate(ts) + ".spacy"
    convert_data(train_path, df=train_data)
    convert_data(dev_path, df=dev_data)

    # train the model: https://spacy.io/usage/training#quickstart
    # python -m spacy train ./inputs/efficiency_cpu_config.cfg --output ./outputs --paths.train ./inputs/train.spacy --paths.dev ./inputs/dev.spacy

