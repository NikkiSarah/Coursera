# install required packages
# install.packages("stargazer")
# install.packages("plyr")
# install.packages("dplyr")
# install.packages("magrittr")
# install.packages("ggplot2")
# install.packages("ggthemes")
# install.packages("gridExtra")
# install.packages("AER")
# install.packages("MASS")
# install.packages("glmnet")
install.packages("simstudy", method="wininet")
# install.packages("randomForest")
install.packages("grf", method = "wininet")
# install.packages("lubridate")

# load required packages
library(ggplot2)
library(ggthemes)
library(scales)
library(gridExtra)
library(lubridate)
library(stargazer)
library(plyr)
library(dplyr)
library(magrittr)
library(ggplot2)
library(ggthemes)
library(AER)
library(MASS)
library(glmnet)
library(simstudy)
library(randomForest)
library(grf)
library(lubridate)

###############################################################

## Controlled / Fixed Effects Regression

# set constants:
# n = number of data points; n_time_periods = years of data;
# n_products = number or products; B = true regression coeff
sim_fixed_effects_df <- function(n = 10 ^ 4,
                                 n_time_periods = 10,
                                 n_products = 5,
                                 B = 2) {
  set.seed(30)
  #set variables; note X depends on fixed effects & other controls
  Time_FE <- paste("Year", rep((year(Sys.Date()) - n_time_periods + 1):
                                 year(Sys.Date()),
                               times = ceiling(n / n_time_periods))[1:n])
  Product_FE <- rep(paste("Product", LETTERS[1:n_products]),
                    each = ceiling(n / n_products))[1:n]
  Customer_Rating <- round(pmax(pmin(rnorm(n = n, mean = 5, sd = 1), 5), 1), 2)
  Customer_Age <- round(rnorm(n = n, mean = 30, sd = 4), 0)
  Total_Purchases <- round(100 * Customer_Rating + rnorm(n = n, mean = 10), 0)
  e3 <- rnorm(n = n, sd = sd(Customer_Rating))
  Customer_Spend <- round(500 + B * Customer_Rating + 10 * Customer_Age + 10 *
                            as.numeric(as.factor(Time_FE)) +
                            20 * as.numeric(as.factor(Product_FE)) + e3, 0)
  dat <- data.frame(
    Customer_Spend,
    Customer_Rating,
    Customer_Age,
    Product_FE,
    Time_FE,
    Total_Purchases
  )
  return(dat)
}

###############################################################

## Regression Discontinuity

# set constants:
# cutoff = number between 0 and 100 to set discontinuity
# mu = mean customer spend; sd = standard dev of customer spend
# treatment_eff = causal impact of intervention point cutoff
sim_reg_discontinuity_df <- function(cutoff = 70,
                                     mu = 20,
                                     sigma = 5,
                                     treatment_eff = 25) {
  set.seed(30)
  dat <- data.frame("Lead_Score" = seq(from = 0, to = 100, by = 1))
  dat$Add_Support <- dat$Lead_Score >= cutoff
  dat$Counterfactual <- dat$Lead_Score * rnorm(n = nrow(dat),
                                               mean = mu,
                                               sd = sigma) / 10
  dat$Customer_Spend[!dat$Add_Support] <- dat$Counterfactual[!dat$Add_Support]
  dat$Customer_Spend[dat$Add_Support] <- dat$Lead_Score[dat$Add_Support] * 
    rnorm(n = sum(dat$Add_Support),
          mean = mu + treatment_eff,
          sd = sigma) / 10
  return(dat)
}

###############################################################

## Difference in Difference

# set constants:
# mu1 = mean of base group / US
# sigma1 = standard dev of base group / US
# mu2 = mean of treatment group / AU
# sigma2 = standard dev of treatment group / AU
# time_change = change in group mean over time
# causal_effect = causal impact of intervention in post period
# and treatment group
sim_diff_in_diff_df <- function(mu1 = 100,
                                mu2 = 200,
                                sigma1 = 25,
                                sigma2 = 25,
                                time_change = 200,
                                causal_effect = 100) {
  set.seed(30)
  dat <- data.frame("Time" = rep(seq(
    from = as.Date("2018-01-01"),
    to = as.Date("2020-01-01"),
    by = "month"), times = 2))
  dat$Period <- ifelse(dat$Time < "2019-01-01",
                       "Pre_Price_Change",
                       "Post_Price_Change")
  dat$Country <- rep(c("US", "AU"), each = nrow(dat) / 2)
  dat$Revenue <- c(
    rnorm(sum(dat$Time < "2019-01-01") / 2, mu1, sigma1),
    rnorm(sum(dat$Time >= "2019-01-01") / 2, mu1 + time_change, sigma1),
    rnorm(sum(dat$Time < "2019-01-01") / 2, mu2, sigma2),
    rnorm(sum(dat$Time >= "2019-01-01") / 2, mu2 + time_change + causal_effect,
          sigma2)
    )
  dat$Counterfactual <- dat$Revenue
  dat$Counterfactual[dat$Period == "Post_Price_Change" & dat$Country == "US"] <-
    rnorm(sum(dat$Time >= "2019-01-01") / 2, mu2 + time_change, sigma2)
  dat$Period <- relevel(factor(dat$Period), ref = "Pre_Price_Change")
  dat$Country <- relevel(factor(dat$Country), ref = "US")
  return(dat)
}

###############################################################

## Instrumental Variable

# set constants:
# n = number of observations
# latent_prob_impact = impact of latent variable on both X
# and Y propensity
# intervention_impact = impact of instrument variable on both X
# propensity (assume impact on Y is zero)
sim_iv_df <- function(n = 10 ^ 4,
                      latent_prob_impact = 0.25,
                      intervention_impact = 0.25) {
  set.seed(30)
  dat <- data.frame("Received_Email" = sample(c(0, 1), size = n, replace = T))
  dat$Unobs_Motivation <- rnorm(n = n)
  dat$Use_Mobile_App <- sapply(1:n, function(r) {
    sample(
      c(0, 1),
      size = 1,
      prob = c(
        0.5 - latent_prob_impact * (dat$Unobs_Motivation[r] > 0) -
          intervention_impact * (dat$Received_Email[r] == 1),
        0.5 + latent_prob_impact * (dat$Unobs_Motivation[r] > 0) +
          intervention_impact * (dat$Received_Email[r] == 1)
      )
    )
  })
  dat$Retention <- sapply(1:n, function(r) {
    sample(
      c(0, 1),
      size = 1,
      prob = c(
        0.5 - latent_prob_impact * (dat$Unobs_Motivation[r] > 0),
        0.5 + latent_prob_impact * (dat$Unobs_Motivation[r] > 0)
      )
    )
  })
  return(dat)
}

###############################################################

## Double Selection

# set constants:
# n = number of observations
# N_Coeff = number of control coefficients to simulate data for
# B = causal impact of treatment on outcome
sim_double_selection_df <- function(n = 10 ^ 3,
                                    N_Coeff = 5 * 10 ^ 2,
                                    B = 2) {
  set.seed(30)
  #mean of regression coefficients for control variables
  beta_C_mu <- 25
  beta_C_sigma <- 10
  #number of nonzero control coefficients
  beta_C_n_zero <- 10
  #simulate control variable values as correlated variables
  C_mu <- rep(1, N_Coeff)
  C_var <- rnorm(N_Coeff, mean = 0, sd = 0.1) ^ 2
  C_rho <- 0.5
  C <- as.data.frame.matrix(
    genCorGen(
      n = n,
      nvars = N_Coeff,
      params1 = C_mu,
      params2 = C_var,
      dist = 'normal',
      rho = C_rho,
      corstr = 'ar1',
      wide = 'True'
    )
  )[, -1]
  #simulate beta coefficients for control variables
  #and set portion of them to zero
  betaC <- rnorm(N_Coeff, mean = beta_C_mu, sd = beta_C_sigma)
  betaC[beta_C_n_zero:N_Coeff] <- 0
  #generate treatment indicator and randomize
  Social_Proof_Variant <- rep(0, n)
  Social_Proof_Variant[0:(n / 2)] <- 1
  Social_Proof_Variant <- sample(Social_Proof_Variant)
  #simulate random noise
  e <- rnorm(n)
  #generate outcome variable
  Customer_Value <- B * Social_Proof_Variant + data.matrix(C) %*% betaC +
    e
  dat <- data.frame(Customer_Value, Social_Proof_Variant, C)
  return(dat)
}

###############################################################

## Causal Forests

# set constants:
# n = number of observations
# N_Coeff = number of control coefficients to simulate data for
# N_groups = number of groups want to estimate causal impact for
sim_causal_forest_df <- function(n = 5 * 10 ^ 3, N_Coeff = 5) {
  set.seed(30)
  N_groups <- 4
  beta <- rep(c(1:N_groups), each = n / N_groups) * 5
  var_group <- factor(beta)
  levels(var_group) <- c("Google", "Instagram", "Twitter", "Bing")
  var_group <- relevel(var_group, ref = "Google")
  C_mu <- rep(0, N_Coeff)
  C_rho <- 0.5
  C_var <- rnorm(N_Coeff, mean = 1, sd = 1) ^ 2
  beta_C_mu_sigma <- 10
  C <- as.data.frame.matrix(
    genCorGen(
      n = n,
      nvars = N_Coeff,
      params1 = C_mu,
      params2 = C_var,
      dist = 'normal',
      rho = C_rho,
      corstr = 'ar1',
      wide = 'True'
    )
  )[, -1]
  betaC <- rnorm(N_Coeff, mean = beta_C_mu_sigma, sd = beta_C_mu_sigma)
  Discount <- rep(0, n)
  Discount[0:(n / 2)] <- 1
  Discount <- sample(Discount)
  e <- rnorm(n)
  Revenue <- 200 + beta * Discount + data.matrix(C) %*% betaC + e
  dat <- data.frame(Revenue, Discount, C,
                    "Registration_Source" = var_group)
  return(dat)
}

###############################################################