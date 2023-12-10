## Task 1: Extract word features and show EDA - inspect, visualise and clean data
# Show a few visualizations like histograms. Describe any data cleaning procedures.
# Based on your EDA, what is your plan of analysis?
#
# Look at online resources on processing raw texts to feature vectors. Many methods process texts
# to matrix form (word embedding), including TF-IDF, GloVe, Word2Vec, etc. Pick a method and process the raw texts to
# word embedding. Briefly explain the method(s) and how they work in your own words. Also, do exploratory data analysis
# such as word statistics and/or visualisation.
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



