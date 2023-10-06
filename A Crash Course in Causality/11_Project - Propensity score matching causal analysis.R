
# Propensity Score Causal Analysis ----------------------------------------

# In this analysis, we are using data from Lalonde (1986) that aimed to evaluate
# the impact of National Supported Work (NSW) Demonstration, a labour training
# program, on post-intervention income levels.

# There are 614 subjects and 10 variables:
# age - age in years
# educ - years of schooling
# black - indicator variable for African-Americans
# hispan - indicator variable for hispanics
# married - indicator variable for marital status
# nodegree - indicator variable for high school diploma
# re74 - real earnings in 1974
# re75 - real earnings in 1975
# re78 - real earnings in 1978
# treat - indicator variable for treatment status (whether or not the subject
# received labour training)

# Outcome: re78
# Treatment: treat
# Potential confounders: age, educ, black, hispan, married, nodegree, re74, re75

# Load Packages and Data --------------------------------------------------

install.packages(c("tableone", "Matching", "MatchIt"), method="wininet")

library(dplyr)
library(tableone)
library(Matching)
library(MatchIt)
library(tidyr)

data("lalonde")
lalonde <- lalonde %>% 
  mutate(black = if_else(race == "black", 1, 0),
         hispan = if_else(race == "hispan", 1, 0)) %>% 
  dplyr::select(-race) %>% 
  dplyr::select(re78, treat, everything())
glimpse(lalonde)

covariates <- names(lalonde)[3:10]

# create a pre-matching table 1
table1_init <- CreateTableOne(vars = covariates, strata = "treat",
                              data = lalonde, test = FALSE)
print(table1_init, smd = TRUE)
# every confounder except for educ shows some degree of imbalance

y_treated <- lalonde$re78[lalonde$treat == 1]
y_control <- lalonde$re78[lalonde$treat == 0]
raw_mean_diff <- mean(y_treated) - mean(y_control)
raw_mean_diff

# Propensity Score Modelling ----------------------------------------------

# remember to exclude the outcome variable from the model
ps_mod <- glm(treat ~ .-re78, data = lalonde, family = "binomial")
summary(ps_mod)

# estimate the propensity score (the probability of being treated) for each
# subject in the data
pscores <- ps_mod$fitted.values
min(pscores)
max(pscores)

# match treatment and control subjects based on the propensity scores without
# replacement and without a caliper
set.seed(931139)

ps_match <- Match(Tr = lalonde$treat, X = pscores, M = 1, replace = FALSE)
matched <- lalonde[unlist(ps_match[c("index.treated", "index.control")]), ]

table1_matched <- CreateTableOne(vars = covariates, strata = "treat",
                                 data = matched, test = FALSE)
print(table1_matched, smd = TRUE)
# the matching worked reasonably well (nodegree, black and hisplan still show
# signs of imbalance) and all treated observations were retained.

# match treatment and control subjects with a caliper of 0.1
set.seed(931139)

ps_match <- Match(Tr = lalonde$treat, X = pscores, M = 1, replace = FALSE,
                  caliper = 0.1)
matched <- lalonde[unlist(ps_match[c("index.treated", "index.control")]), ]

table1_matched <- CreateTableOne(vars = covariates, strata = "treat",
                                 data = matched, test = FALSE)
print(table1_matched, smd = TRUE)
# with the addition of a caliber, some treated observations were dropped, but
# now only re75 shows signs of imbalance

# note how the mean difference has changed sign (from a negative to positive
# amount)
y_treated <- matched$re78[matched$treat == 1]
y_control <- matched$re78[matched$treat == 0]
matched_mean_diff <- mean(y_treated) - mean(y_control)
matched_mean_diff

diff_y <- y_treated - y_control
t.test(diff_y)
# the t-test indicates that despite the change in sign, there is no evidence of
# a treatment effect i.e. labour work program participation did not affect
# 1978 income
