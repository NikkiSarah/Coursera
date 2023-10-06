
# IPTW Causal Analysis ----------------------------------------------------

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

install.packages(c("tableone", "Matching", "MatchIt", "ipw"), method="wininet")

library(dplyr)
library(tableone)
library(Matching)
library(MatchIt)
library(ipw)
library(survey)
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

# IPTW Modelling ----------------------------------------------------------

# remember to exclude the outcome variable from the model
ps_mod <- glm(treat ~ .-re78, data = lalonde, family = "binomial")
summary(ps_mod)

# extract the propensity scores
pscores <- predict(ps_mod, type = "response")

# create the iptw weights
weights = if_else(lalonde$treat == 1, 1/pscores, 1/(1-pscores))
min(weights)
max(weights)

# apply weights to the data
weighted_lalonde <- svydesign(ids = ~1, data = lalonde, weights = ~weights)

# create a weighted table 1
table1_weighted <- svyCreateTableOne(vars = covariates, strata = "treat",
                                     data = weighted_lalonde, test = FALSE)
# note that the balance is better than the raw data, but still not ideal
print(table1_weighted, smd = TRUE)

# fit the marginal structural model (risk difference)
lalonde$weights <- weights

msm <- (svyglm(re78 ~ treat,
               design = svydesign(~1, weights = ~weights, data = lalonde)))
# extract the model coefficients for a causal risk difference
coef(msm)
# extract the 95% confidence interval
confint(msm)

# IPTW Modelling with Weight Truncation -----------------------------------

lalonde2 <- lalonde %>% dplyr::select(-weights)
weight_model <- ipwpoint(exposure = treat, family = "binomial",
                         link = "logit", denominator = ~.-re78, data = lalonde2,
                         trunc = 0.01)
summary(weight_model$weights.trunc)
lalonde2$truncated_weights <- weight_model$weights.trunc

# fit the marginal structural model (risk difference)
msm <- (svyglm(re78 ~ treat,
               design = svydesign(~1, weights = ~truncated_weights,
                                  data = lalonde2)))
# extract the model coefficients
coef(msm)
# extract the 95% confidence interval
confint(msm)