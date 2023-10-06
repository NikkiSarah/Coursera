
# Matched Causal Analysis -------------------------------------------------

# This script demonstrates an example of how to carry out a matched covariate
# causal analysis in R.

# Data: Right heart catheterisation (RHC)
# ICU patients in 5 hospitals
# Treatment: RHC (Yes/No)
# Outcome: Death (Yes/No)
# Confounders: Demographics, insurance, disease diagnoses etc.
# 2,184 treated and 3,551 controls

# Load Packages and Data --------------------------------------------------

install.packages(c("tableone", "Matching"), method="wininet")

library(dplyr)
library(janitor)
library(readr)
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
  select(cat1, sex, death, age, swang1, meanbp1) %>% 
  mutate(female = if_else(sex == "Female", 1, 0),
         died = if_else(death == "Yes", 1, 0),
         treatment = if_else(swang1 == "RHC", 1, 0)) %>% 
  rename(meanbp = meanbp1) %>% 
  select(-c(sex, death, swang1))

cat1_dummies <- as.data.frame(model.matrix(~rhc$cat1 - 1)) %>% clean_names()
names(cat1_dummies) <- sub(pattern = "rhc_cat1", replacement = "", names(cat1_dummies))

rhc <- rhc %>% 
  bind_cols(cat1_dummies) %>% 
  select(-cat1) %>% 
  # reorder variables to outcome, treatment, confounders
  select(died, treatment, everything()) %>% 
  # copd will be the baseline for the cat1 values
  select(-copd)
glimpse(rhc)

covariates <- names(rhc)[3:13]

# create a pre-matching table 1
# recall that we're particularly concerned about SMDs > 0.1. In this example,
# variables of concern are meanbp1, cirrhosis, coma and mosf_w_sepsis.
table1_init <- CreateTableOne(vars = covariates, strata = "treatment",
                              data = rhc, test = FALSE)
print(table1_init, smd = TRUE)

# Greedy Matching on Mahalanobis Distance ---------------------------------

# calculate distances between every covariate and match on the entire set
# M = 1 indicates pair/one-on-one matching
greedy_match <- Match(Tr = rhc$treatment, M = 1, X = rhc[covariates])

# note how the number of observations reduced from 5,735 to 4,372, but when
# looking at the table 1, all treated observations had a match in the control
# group (2,186 treated)
matched <- rhc[unlist(greedy_match[c("index.treated", "index.control")]), ]

table1_matched <- CreateTableOne(vars = covariates, strata = "treatment",
                                 data = matched, test = FALSE)
# matching process worked well in this case as there is great balance
print(table1_matched, smd = TRUE)

# use a paired t-test to obtain a causal risk difference
# result indicates that there is a treatment effect (p-value approx 0)
# 0.045 is the average risk difference, which is the difference in the probability
# of death if everyone received the treatment vs if no-one did i.e. higher risk of
# death in the treatment group
# 95% CI indicates the plausible range of the true risk difference (here 0.02 to 0.07)
y_treated <- matched$died[matched$treatment == 1]
y_control <- matched$died[matched$treatment == 0]
diff_y <- y_treated - y_control
t.test(diff_y)

# carry out a McNemar test
# recall that this shows the outcome for each pair so 994 pairs had an outcome of
# 1 for both groups, 305 pairs had an outcome of 0 for both groups
# in the discordant pairs, there are 493 pairs where the treated subject died
# whereas the control did not, and only 394 pairs in which the control subject
# died but the treated subject did not. As the bigger number is the one in which
# the treated subjects died, this provides further evidence that the treatment
# group is of higher risk of death.
table(y_treated, y_control)

# again, get confirmation that there is a treatment effect
mcnemar.test(table(y_treated, y_control))

# Discussion --------------------------------------------------------------

# 1. If we wanted a causal risk ratio or causal odds ratio, we could use GEE
# with a log or logit link, respectively.
# 2. In practice, a larger number of covariates would be used than in this
# example. The number of covariates was restricted for simplicity.
# 3. There are other R packages available for matching, such as the rcbalance
# package (which is also more sophisticated). For example, if you required a
# fine balance constraint or wanted optimal matching.
