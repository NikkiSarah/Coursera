
# IPTW Causal Analysis ----------------------------------------------------

# This script demonstrates an example of how to carry out an Inverse
# Probability of Treatment Weighting causal analysis in R.

# Data: Right heart catheterisation (RHC)
# ICU patients in 5 hospitals
# Treatment: RHC (Yes/No)
# Outcome: Death (Yes/No)
# Confounders: Demographics, insurance, disease diagnoses etc.
# 2,184 treated and 3,551 controls

# Load Packages and Data --------------------------------------------------

install.packages(c("tableone", "ipw"), method="wininet")

library(dplyr)
library(janitor)
library(readr)
library(ggplot2)

library(tableone)
library(ipw)
library(sandwich)
library(survey) # to get weighted estimators

# root_dir <- r"{J:/AID/Business Functions/Data and Analytical Services/Impact Evaluation Research/Training_Upskilling Materials/July 2023 A Crash Course in Causality (R)}"
# setwd(root_dir)

# rhc <- read_csv(paste(root_dir, "rhc.csv", sep = "\\"))
rhc <- read_csv("rhc.csv")
glimpse(rhc)

# select variables necessary for analysis and convert any categoricals to their
# numeric equivalent (some models can natively handle categoricals and some
# can't, so whether the latter transformation is useful is dependent on the
# approach)
rhc <- rhc %>% 
  dplyr::select(cat1, sex, death, age, swang1, meanbp1) %>% 
  mutate(female = if_else(sex == "Female", 1, 0),
         died = if_else(death == "Yes", 1, 0),
         treatment = if_else(swang1 == "RHC", 1, 0)) %>% 
  rename(meanbp = meanbp1) %>% 
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

# view an unbalanced table 1
covariates <- names(rhc)[3:13]
table1 <- CreateTableOne(vars = covariates, strata = "treatment", data = rhc,
                         test = FALSE)
print(table1, smd = TRUE)

# Fit a Propensity Score Model --------------------------------------------

# train a propensity score model
# here we've used a logistic regression, but you can use any binary
# classification algorithm
ps_model <- glm(treatment ~.-died, data = rhc, family = "binomial")
# summary statistics provide some insight into what covariates are most 
# predictive of treatment i.e. what's different about subjects receiving
# treatment vs subjects in the control group
# also to check that coefficients are in the expected direction or more generally
# that the model makes sense
summary(ps_model)
# extract the propensity scores
pscores <- predict(ps_model, type = "response")

# compare the distributions of the propensity scores pre-weighting
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

# Create Weights and Check Balance ----------------------------------------

# create the weights
weights = if_else(rhc$treatment == 1, 1/pscores, 1/(1-pscores))

# apply weights to the data
# there is more than one way to do this step
weighted_rhc <- svydesign(ids = ~1, data = rhc, weights = ~weights)

# create a weighted table 1
covariates <- names(rhc)[3:13]
table1_weighted <- svyCreateTableOne(vars = covariates, strata = "treatment",
                                     data = weighted_rhc, test = FALSE)
# ignore the standard deviations (in brackets) as they're based on the weighted
# sample sizes, which aren't the true sample sizes. The means, which are the
# weighted means of the pseudo-population are correct and should be paid
# attention to.
# note that there is excellent balance here.
print(table1_weighted, smd = TRUE)

# get a weighted mean directly for a covariate
# these should match up exactly with the table 1 results
for (cov in covariates) {
  weight = mean(weights[rhc$treatment == 1]*rhc[[cov]][rhc$treatment == 1]) /
    (mean(weights[rhc$treatment == 1]))
  
  print(paste0(cov, ": ", round(weight, 4)))
}

# Fit a Marginal Structural Model -----------------------------------------

# E(Y_i^a) = g^-1(\psi_0 + \psi_1*a)
# A is the treatment
# Y is the outcome
# g() is the link function
# we'll first use a log link to get a causal relative risk and then an identity
# link to get a causal risk difference

# get a causal relative risk using a weighted GLM
# expit <- function(x) {1/(1+exp(-x)) }
# logit <- function(p) {log(p)-log(1-p)}
glm_model <- glm(died ~ treatment, weights = weights, data = rhc,
                 family = quasibinomial(link = log))
summary(glm_model)

# extract the model coefficients
beta_iptw <- coef(glm_model)

# use asymptotic (sandwich) variance to properly account for the weighting
# i.e. the weights inflate the sample size
se <- sqrt(diag(vcovHC(glm_model, type = "HC0")))

# get point estimates and CI for relative risk
causal_rr <- exp(beta_iptw[2])
lcl <- exp(beta_iptw[2] - 1.96*se[2])
ucl <- exp(beta_iptw[2] + 1.96*se[2])
c(lcl, causal_rr, ucl)
# these are the point estimates plus the 95% confidence interval
# a value greater than 1 indicates a higher risk of death for the treated group


# get a causal risk difference using a weighted GLM with an identity link
# the model being fitted here is E(Y^a) = \psi_0 + \psi_1 * a
glm_model <- glm(died ~ treatment, weights = weights, data = rhc,
                 family = quasibinomial(link = "identity"))
summary(glm_model)

beta_iptw <- coef(glm_model)
se <- sqrt(diag(vcovHC(glm_model, type = "HC0")))

# get point estimates and CI for a risk difference
# note that there's no need for exponentiation
causal_rd <- beta_iptw[2]
lcl <- beta_iptw[2] - 1.96*se[2]
ucl <- beta_iptw[2] + 1.96*se[2]
c(lcl, causal_rd, ucl)


# observe the impact of truncating the weights at 10 on the causal risk
# difference point estimate and confidence interval
truncated_weights <- if_else(weights > 10, 10, weights)
summary(truncated_weights)

glm_model <- glm(died ~ treatment, weights = truncated_weights, data = rhc,
                 family = quasibinomial(link = "identity"))
summary(glm_model)

beta_iptw <- coef(glm_model)
se <- sqrt(diag(vcovHC(glm_model, type = "HC0")))

# get point estimates and CI for a risk difference
# note that there's no need for exponentiation
causal_rd <- beta_iptw[2]
lcl <- beta_iptw[2] - 1.96*se[2]
ucl <- beta_iptw[2] + 1.96*se[2]
c(lcl, causal_rd, ucl)


# fit the same models using the IPW package
# ipwpoint needs a plain dataframe, not a tibble
rhc_df <- as.data.frame(rhc)

# what we're really doing here is specifying the propensity score model
# denominator is the denominator of the weights, which is really the propensity
# score model inputs
weight_model <- ipwpoint(exposure = treatment, family = "binomial",
                         link = "logit", denominator = ~.-died, data = rhc_df)
# numeric summary of the weights
summary(weight_model$ipw.weights)
# save the weights
rhc$weights <- weight_model$ipw.weights

# plot the weight distribution
ipwplot(weights = weight_model$ipw.weights, logscale = FALSE, main = "weights",
        xlim = c(0, 22))

# fit the marginal structural model (risk difference)
# the primary advantage over the plain glm is that it will give you the correct
# variance estimators i.e. will give the robust sandwich estimator
               # this part is the msm
msm <- (svyglm(died ~ treatment,
               # this part indicates that weights to be used
               design = svydesign(~1, weights = ~weights, data = rhc)))
# extract the model coefficients
# these show an identity link situation i.e. should be exactly the same as the
# causal risk difference estimated earlier
coef(msm)
# extract the 95% confidence interval
confint(msm)


# ipw has a built in weight-truncation function
# for example, this model truncates the weights at the 1st and 99th percentiles
# if you wanted to truncate at a particular value, you'd have to check the weight
# distribution and identify what percentile that weight corresponded to
weight_model <- ipwpoint(exposure = treatment, family = "binomial",
                         link = "logit", denominator = ~.-died, data = rhc_df,
                         trunc = 0.01)
summary(weight_model$weights.trunc)
rhc$truncated_weights <- weight_model$weights.trunc

ipwplot(weights = weight_model$weights.trunc, logscale = FALSE,
        main = "truncated weights", xlim = c(0, 10))

# fit the marginal structural model (risk difference)
msm <- (svyglm(died ~ treatment,
               design = svydesign(~1, weights = ~truncated_weights,
                                  data = rhc)))
# extract the model coefficients
coef(msm)
# extract the 95% confidence interval
confint(msm)

# the truncation didn't have a material impact on the results, but that was
# largely expected given that there weren't any extreme weights present
