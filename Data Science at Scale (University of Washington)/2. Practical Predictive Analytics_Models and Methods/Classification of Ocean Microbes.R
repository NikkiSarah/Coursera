# Classification of Ocean Microbes ----------------------------------------
#
# You will be working with data from the SeaFlow environmental flow cytometry
# instrument.
# 
# A flow cytometer delivers a flow of particles through capilliary. By shining
# lasers of different wavelengths and measuring the absorption and refraction
# patterns, you can determine how large the particle is and some information
# about its color and other properties, allowing you to detect it.
# 
# The technology was developed for medical applciations, where the particles
# were potential pathogens in, say, serum, and the goal was to give a diagnosis.
# But the technology was adapted for use in environmental science to understand
# microbial population profiles.
# 
# The SeaFlow instrument, developed by the Armbrust Lab at the University of
# Washington, is unique in that it is deployed on research vessels and takes
# continuous measurements of population profiles in the open ocean.
# 
# While there are a number of challenging analytics tasks associated with this
# data, a central task is classification of particles. Based on the optical
# measurements of the particle, it can be identified as one of several populations.
#
# Dataset -----------------------------------------------------------------
#
# You have been provided with a dataset that represents a 21 minute sample from
# the vessel in a file seaflow_21min.csv. This sample has been pre-processed to
# remove the calibration "beads" that are passed through the system for
# monitoring, as well as some other particle types.
# 
# The columns of this dataset are as follows:
# - file_id: The data arrives in files, where each file represents a
# three-minute window; this field represents which file the data came from.
# The number is ordered by time, but is otherwise not significant.
# 
# - time: This is an integer representing the time the particle passed through
# the instrument. Many particles may arrive at the same time; time is not a key
# for this relation.
# 
# - cell_id: A unique identifier for each cell WITHIN a file.
# (file_id, cell_id) is a key for this relation.
# 
# - d1, d2: Intensity of light at the two main sensors, oriented
# perpendicularly. These sensors are primarily used to determine whether the
# particles are properly centered in the stream. Used primarily in
# preprocesssing; they are unlikely to be useful for classification.
# 
# - fsc_small, fsc_perp, fsc_big: Forward scatter small, perpendicular, and big.
# These values help distingish different sizes of particles.
# 
# - pe: A measurement of phycoerythrin fluorescence, which is related to the
# wavelength associated with an orange color in microorganisms.
# 
# - chl_small, chl_big: Measurements related to the wavelength of light
# corresponding to chlorophyll.
# 
# - pop: This is the class label assigned by the clustering mechanism used in
# the production system. It can be considered "ground truth" for the purposes
# of the assignment, but note that there are particles that cannot be
# unambiguously classified, so you should not aim for 100% accuracy. The values
# in this column are crypto, nano, pico, synecho, and ultra.
# 
# Step 1: Read and summarise the data -------------------------------------
library(tidyverse)

data <- read_csv("seaflow_21min.csv")
summary(data)
head(data)

table(data$pop)

# There are 18,146 particles labelled "synecho" in the data.
# The 3rd quantile of the variable `fsc_small` is 39,184.

# Step 2: Split the data into training and test sets ----------------------

library(tidymodels)

data$pop <- as.factor(data$pop)

data_split <- initial_split(data, prop = 0.8, strata = pop)
train_data <- training(data_split)
test_data <- testing(data_split)

# The mean of the variable `time` in the training set is 

# Step 3: Plot the data ---------------------------------------------------

data %>% 
  ggplot(aes(x = pe, y = chl_small, colour = pop)) +
  geom_point(shape = 1) +
  labs(x = "phycoerythrin fluorescence measurement",
       y = "small chlorophyll measurement") +
  scale_colour_discrete(name = "Microbial particle type") +
  theme_classic()

# In the plot, the 'ultra' particles are somewhat mixed in with the 'pico' and
# 'nano' particle populations.

# Step 4: Train and evaluate a decision tree ------------------------------

library(rpart.plot)

model_formula <- pop ~ fsc_small + fsc_perp + chl_small + pe + chl_big + chl_small

dt_spec <- decision_tree() %>% 
  set_mode("classification") %>% 
  set_engine("rpart")

dt_mod <- dt_spec %>% fit(model_formula, data = train_data)
print(dt_mod)

rpart.plot(dt_mod$fit)

# From the plot, the model is incapable of recognising the 'crypto' particle
# population.

# The threshold value for the `pe` column in the decision tree model is 5,004.

# The variables most important in predicting the class population are `pe` and
# `chl_small`.

dt_test_preds <- predict(dt_mod, test_data)
dt_test_accuracy <- sum(dt_test_preds$.pred_class == test_data$pop) / nrow(test_data)
dt_test_accuracy

# Step 5: Train and evaluate a random forest ------------------------------

rf_spec <- rand_forest() %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

rf_spec <- rf_spec %>% set_args(importance = "impurity")

rf_mod <- rf_spec %>% fit(model_formula, data = train_data)
print(rf_mod)

rf_test_preds <- predict(rf_mod, test_data)
rf_test_accuracy <- sum(rf_test_preds$.pred_class == test_data$pop) / nrow(test_data)
rf_test_accuracy

# Random forests can automatically generate an estimate of variable importance
# during training by permuting the values in a given variable and measuring the
# effect on classification. If scrambling the values has little effect on the
# model's ability to make predictions, then the variable must not be very
# important.
# 
# A random forest can obtain another estimate of variable importance based on
# Gini impurity. The function importance(model) prints the mean decrease in gini
# importance for each variable. The higher the number, the more the gini
# impurity score decreases by branching on this variable, indicating that the
# variable is more important.

importance(rf_mod$fit)

# The two variables most important in predicting the class population based on
# gini impurity are `pe` and `chl_small`.

# Step 6: Train and evaluate a support vector machine ---------------------

svm_spec <- svm_rbf() %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

svm_recipe <- recipe(model_formula, data = train_data) %>% 
  step_normalize(all_predictors())
svm_recipe
  
svm_mod <- svm_spec %>% fit(model_formula, data = train_data)
print(svm_mod)

svm_test_preds <- predict(svm_mod, test_data)
svm_test_accuracy <- sum(svm_test_preds$.pred_class == test_data$pop) / nrow(test_data)
svm_test_accuracy

# Step 7: Construct confusion matrices ------------------------------------

table(pred = dt_test_preds$.pred_class, true = test_data$pop)

table(pred = rf_test_preds$.pred_class, true = test_data$pop)

table(pred = svm_test_preds$.pred_class, true = test_data$pop)

# The most common error made by the models is `ultra` particles being mistaken
# for `pico` particles. (The next most common error is `nano` particles being
# mistaken for `ultra` particles.)

# Step 8: Sanity check the data -------------------------------------------

# As a data scientist, you should never trust the data, especially if you did
# not collect it yourself. There is no such thing as clean data. You should
# always be trying to prove your results wrong by finding problems with the
# data. Richard Feynman calls it "bending over backwards to show how you're
# maybe wrong." This is even more critical in data science, because almost by
# definition you are using someone else's data that was collected for some other
# purpose rather than the experiment you want to do. So of course it's going to
# have problems.
# 
# The measurements in this dataset are all supposed to be continuous (fsc_small,
# fsc_perp, fsc_big, pe, chl_small, chl_big), but one is not. Figure out which
# field is corrupted.

long_data <- tidyr::pivot_longer(data, cols = c("fsc_small", "fsc_perp",
                                                "fsc_big", "pe", "chl_small",
                                                "chl_big"))

histograms <- ggplot(long_data, aes(x = value)) +
  geom_histogram(bins = 30, fill = "steelblue", colour = "white") +
  facet_wrap(~ name, scales = "free", ncol = 3) +
  labs(x = NULL, y = NULL) +
  theme_classic()
histograms

# The histograms indicate that the problematic variable is `fsc_big`

# There is more subtle issue with data as well. Plot time vs. chl_big, and you
# will notice a band of the data looks out of place. This band corresponds to
# data from a particular file for which the sensor may have been miss-calibrated.
# Remove this data from the dataset by filtering out all data associated with
# file_id 208, then repeat the experiment for all three methods, making sure to
# split the dataset into training and test sets after filtering out the bad
# data.

data$file_id <- as.factor(data$file_id)

data %>% 
  ggplot(aes(x = time, y = chl_big, colour = file_id)) +
  geom_point(shape = 1) +
  labs(x = "time",
       y = "big chlorophyll measurement") +
  theme_classic()

data2 <- data %>% 
  filter(file_id != 208)

data_split <- initial_split(data2, prop = 0.8, strata = pop)
train_data <- training(data_split)
test_data <- testing(data_split)


dt_mod <- dt_spec %>% fit(model_formula, data = train_data)
print(dt_mod)

rpart.plot(dt_mod$fit)

dt_test_preds <- predict(dt_mod, test_data)
dt_test_accuracy2 <- sum(dt_test_preds$.pred_class == test_data$pop) / nrow(test_data)
dt_test_accuracy2


rf_mod <- rf_spec %>% fit(model_formula, data = train_data)
print(rf_mod)

rf_test_preds <- predict(rf_mod, test_data)
rf_test_accuracy2 <- sum(rf_test_preds$.pred_class == test_data$pop) / nrow(test_data)
rf_test_accuracy2


svm_mod <- svm_spec %>% fit(model_formula, data = train_data)
print(svm_mod)

svm_test_preds <- predict(svm_mod, test_data)
svm_test_accuracy2 <- sum(svm_test_preds$.pred_class == test_data$pop) / nrow(test_data)
svm_test_accuracy2

svm_test_accuracy2 - svm_test_accuracy
# After removing the data associated with file_id 208, the accuracy of the svm
