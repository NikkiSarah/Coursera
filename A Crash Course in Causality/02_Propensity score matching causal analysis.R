
# Matched Causal Analysis -------------------------------------------------

# This script demonstrates an example of how to carry out a matched propensity
# score analysis in R.

# Data: Right heart catheterisation (RHC)
# ICU patients in 5 hospitals
# Treatment: RHC (Yes/No)
# Outcome: Death (Yes/No)
# Confounders: Demographics, insurance, disease diagnoses etc.
# 2,184 treated and 3,551 controls

# Load Packages and Data --------------------------------------------------

install.packages(c("tableone", "Matching", "MatchIt"), method="wininet")

library(dplyr)
library(janitor)
library(readr)
library(ggplot2)
library(MatchIt)
library(tableone)
library(Matching)

root_dir <- r"{J:/AID/Business Functions/Data and Analytical Services/Impact Evaluation Research/Training_Upskilling Materials/July 2023 A Crash Course in Causality (R)}"
setwd(root_dir)

rhc <- read_csv(paste(root_dir, "rhc.csv", sep = "\\"))
glimpse(rhc)

# select variables necessary for analysis and convert any categoricals to their
# numeric equivalent (some models can natively handle categoricals and some
# can't, so whether the latter transformation is useful is dependent on the
# approach)
rhc <- rhc %>% 
  dplyr::select(cat1, sex, death, age, swang1, meanbp1, aps1) %>% 
  mutate(female = if_else(sex == "Female", 1, 0),
         died = if_else(death == "Yes", 1, 0),
         treatment = if_else(swang1 == "RHC", 1, 0)) %>% 
  rename(meanbp = meanbp1,
         aps = aps1) %>% 
  dplyr::select(-c(sex, death, swang1))

cat1_dummies <- as.data.frame(model.matrix(~rhc$cat1 - 1)) %>% clean_names()
names(cat1_dummies) <- sub(pattern = "rhc_cat1", replacement = "", names(cat1_dummies))

rhc <- rhc %>% 
  bind_cols(cat1_dummies) %>% 
  dplyr::select(-cat1) %>% 
  # reorder variables to outcome, treatment, confounders
  dplyr::select(died, treatment, everything()) %>% 
  # copd will be the baseline for the cat1 values
  dplyr::select(-copd)
glimpse(rhc)

# Propensity Score Modelling ----------------------------------------------

# train a propensity score model
# here we've used a logistic regression, but you can use any binary
# classification algorithm
ps_model <- glm(treatment ~.-died, data = rhc, family = "binomial")
# summary statistics provide some insight into what covariates are most 
# predictive of treatment i.e. what's different about subjects receiving
# treatment vs subjects in the control group
# could also check hypotheses/assumptions (such as if you knew certain variables
# were associated with a much higher probability of treatment)
summary(ps_model)
# extract the propensity scores
pscores <- ps_model$fitted.values

# compare the distributions of the propensity scores pre-matching
plot_data <- rhc$treatment %>%
  bind_cols(pscores) %>% 
  rename(treatment = `...1`,
         pscores = `...2`) %>% 
  mutate(treatment = as.factor(treatment))

mu <- plyr::ddply(plot_data, "treatment", summarise, grp.mean = mean(pscores))

plot_data %>% 
  ggplot(aes(x = pscores, colour = treatment, fill = treatment)) +
  geom_histogram(bins = 50, alpha = 0.5, position = "dodge") +
  geom_vline(data = mu, aes(xintercept = grp.mean, colour = treatment),
             linetype = "dashed", linewidth = 1) +
  theme_classic()
# note the overlap between the distributions across almost the entire range of
# propensity scores i.e. no real cause for concern

# Matching ----------------------------------------------------------------

# note that with this package, you don't actually have to fit the propensity
# score model first, it'll do it for you
# matching here will use the nearest-neighbour method (i.e. greedy)
# essentially will calculate the propensity score for each observation and then
# match pairs based on these estimated scores
match_out <- matchit(treatment ~.-died, data = rhc, method = "nearest")
summary(match_out)

# in-built plots to check balance
plot(match_out, type = "jitter")
# no unmatched treated observations in this case (that row is empty)
# check that the second and third row (matched treated and control)
# distributions look similar

plot(match_out, type = "hist")
# note how the shape of the matched distributions are more similar, whereas the
# raw treated has a slight left skew, and the raw control is more obviously
# right skewed (which is why more observations at the lower end of the scale
# ended up unmatched)

# Matching on logit(propensity score) without a caliper -------------------

ps_match <- Match(Tr = rhc$treatment, M = 1, X = log(pscores), replace = FALSE)
matched <- rhc[unlist(ps_match[c("index.treated", "index.control")]), ]

covariates <- names(rhc)[3:14]
table1_matched <- CreateTableOne(vars = covariates, strata = "treatment",
                              data = matched, test = FALSE)
print(table1_matched, smd = TRUE)
# all treated subjects were matched with a contol
# some of the standardised differences are bigger than we'd like, but not overly
# so - so not totally satisfied with the matching process

# Matching on logit(propensity score) with a caliper ----------------------

# a caliper of 0.2 means 0.2 standard deviation units, which means 0.2 times the
# standard deviation of the logit of the propensity score as the matching is on
# the logit of the propensity score and not the propensity score directly
ps_match <- Match(Tr = rhc$treatment, M = 1, X = log(pscores), replace = FALSE,
                  caliper = 0.2)
matched <- rhc[unlist(ps_match[c("index.treated", "index.control")]), ]

table1_matched <- CreateTableOne(vars = covariates, strata = "treatment",
                                 data = matched, test = FALSE)
print(table1_matched, smd = TRUE)
# 1,908 of the treated subjects are matched (276 dropped) - smaller number of
# matches as not allowing bad matches - the trade-off is less bias at the
# expense of higher efficiency/variance
# all standardised differences are below the 0.1 threshold

# Outcome Analysis --------------------------------------------------------

# use a paired t-test to obtain a causal risk difference
# result indicates that there is a treatment effect (p-value > 0.05)
# 0.03 is the average risk difference, which is the difference in the
# probability of death if everyone received the treatment vs if no-one did
# i.e. higher risk of death in the treatment group
# 95% CI indicates the plausible range of the true risk difference
# (here 0.002 to 0.06)
# (not shown here) the results with and without the caliper were very similar
y_treated <- matched$died[matched$treatment == 1]
y_control <- matched$died[matched$treatment == 0]
diff_y <- y_treated - y_control
t.test(diff_y)
