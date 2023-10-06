
# IV Causal Analysis ------------------------------------------------------

# This script demonstrates an example of how to carry out an Instrumental
# Variable causal analysis in R.

# Data: Geographic proximity and the return to schooling
# Instrument: Subject grew up near a 4 year college (Yes/No)
# Treatment: Years of education
# Outcome: Income
# Confounders: Parents years of education, region of country, age, race,
#              IQ score etc

# Study Motivation
# More schooling is associated with higher income, but is it due to the fact
# that people with more schooling are different in other ways i.e. beyond
# education.
# Confounding - measured and unmeasured - is a real concern.
# Proposal: Living close to a 4-year college 'encourages' a person to stay in
# school longer, and arguably not directly connected to income once certain
# other variables are controlled for.


# Load Packages and Data --------------------------------------------------

install.packages(c("ivmodel", "ivreg", "ivtools"), method="wininet")

library(dplyr)
library(janitor)
library(readr)
library(ggplot2)

library(ivreg)
library(sandwich)

card_data <- read_csv("card.csv")
glimpse(card_data)

# view summary statistics of the main variables
mean(card_data$nearc4)
# about 68.2% of subjects were 'encouraged'

par(mfrow = c(1,2))
hist(card_data$lwage)
hist(card_data$educ)

# note that income is often log-transformed due to a skewed distribution
# reasonable amount of variability in the amount/years of education
# the education spikes likely correspond to finishing high-school and finishing
# an undergraduate degree

# Estimate ITT and CACE ---------------------------------------------------

# convert education to a binary variable
# could have left it as a continuous variable, but did so to be able to estimate
# the proportion of compliers and make this analysis analogous to a randomised
# trial
card <- card_data %>% 
  mutate(educ_gt_12 = if_else(educ > 12, TRUE, FALSE)) %>% 
  select(nearc4, educ_gt_12, lwage, exper, reg661:reg668)
educ_gt_12 <- card$educ_gt_12

# estimate the proportion of 'compliers'
# in this context, compliers have more than 12 years of schooling if they lived
# near a 4-year college and 12 or fewer years of schooling if they didn't.
# the instrument isn't particularly strong, but also not so weak that we're
# alarmed and need to re-consider our choice.
prop_comp <- mean(educ_gt_12[card_data$nearc4 == 1]) -
  mean(educ_gt_12[card_data$nearc4 == 0])
print(prop_comp)

# estimate the intention to treat effect
# this is the causal effect of encouragement or in this case, the causal effect
# of living near a 4-year college on wages.
# log of wages tends to be higher amongst people who live near a 4-year college
itt_effect <- mean(card$lwage[card$nearc4 == 1]) - 
  mean(card$lwage[card$nearc4 == 0])
print(itt_effect)

# estimate the CACE
# larger than the ITT effect, which is expected as long as the no-defiers
# assumption is made. (In this situation, a defier is someone who lives near a
# 4-year college and doesn't have more than 12 years of schooling, or doesn't
# live near a 4-year college and does.)
cace <- itt_effect / prop_comp
print(cace)

# Estimate ITT and CASE via 2SLS ------------------------------------------

# stage 1: regress A on Z (treatment on the instrument)
s1_lm <- lm(educ_gt_12 ~ nearc4, data = card)
# get the predicted values of A given Z
# subjects living near a 4-year college have a predicted probability of getting
# 12 or more years of schooling of 0.544 and subjects living away from a
# 4-year college have a predicted probability of 0.422
card$pred_A <- predict(s1_lm, type = "response")
table(card$pred_A)

# stage 2: regress Y on the predicted value of A (predicted treatment)
s2_lm <- lm(lwage ~ pred_A, data = card)
summary(s2_lm)
# the coefficient on pred_A is exactly the same as the previous CACE estimate.

# In the case of no covariates, either approach works.

# Estimate ITT and CACE directly ------------------------------------------

iv_model <- ivreg(lwage ~ educ_gt_12 | nearc4, x = TRUE, data = card)
summary(iv_model, vcov = sandwich)  # robust standard errors
# causal effect is the coefficient of the educ_gt_12 variable and again will be
# exactly the same as previous estimates.
# the p-value indicates that there's strong evidence of a causal effect assuming
# the IV assumptions are met.

# Controlling for covariates ----------------------------------------------

# Might argue that the IV assumptions aren't particularly plausible at the
# moment. For example, people living near 4-year colleges might be higher income
# families anyway as the housing could be more expensive.

iv_model_w_covariates <- ivreg(lwage ~ educ_gt_12 + exper + reg661 + reg662 +
                                 reg663 + reg664 + reg665 + reg666 + reg667 + 
                                 reg668 | nearc4 + exper + reg661 + reg662 +
                                 reg663 + reg664 + reg665 + reg666 + reg667 + 
                                 reg668, x = TRUE, data = card)
summary(iv_model_w_covariates, vcov = sandwich)
# still primarily interested in the coefficient on the treatment variable, but
# now controlling for a set of important covariates.
# coefficient now has a slightly different value (1.20 vs 1.28), but is still
# highly significant, which means we're still seeing reasonably strong evidence
# of the number of years of education leading to an increase in wages.
