library(tidyverse)
library(rpart)
library(rpart.plot)
library(caret)

crackers <- read_csv('crackers.csv', col_types = list(col_skip()))

glimpse(crackers)

dt <- rpart(choice ~ ., data = crackers, control = rpart.control(maxdepth = 3))
dt

rpart.plot(dt)

# Rule 1 - root 3292 1500 nabisco (0.06865128 0.54434994 0.31439854 0.07260024)
## Assign every choice to nabisco. Doing so will result in an accuracy of 54.4%
## as there are 1,792 nabisco records in the dataset.

# Rule 2 - price.private >= 59.5 2381  865 nabisco (0.07223856 0.63670727 0.20453591 0.08651827)
## If price.private is greater than 59.5 (which is true for 2,381 rows) then 
## assigning every record to nabisco will result in an accuracy of 63.7%).

# Rule 3 - price.private < 59.5 911  363 private (0.05927552 0.30296378 0.60153677 0.03622393)
## If price.private is less than 59.5 (which is true for 911 rows) then 
## assiging every record to private will result in an accuracy of 60.1%).

ind <- sample(3, nrow(crackers), replace = TRUE, prob = c(0.7, 0.15, 0.15))
train <- crackers[ind == 1, ]
val = crackers[ind == 2, ]
test = crackers[ind == 3, ]

dt <- rpart(choice ~ ., data = train, control = rpart.control(maxdepth = 3))

preds_train <- predict(dt, train, type = 'class')
confusionMatrix(preds_train, factor(train$choice))

preds_val <- predict(dt, val, type = 'class')
confusionMatrix(preds_val, factor(val$choice))

# Overall accuracy for the training model is 67.4% and for the validation model
# it's 69.4%. This indicates that there is little sign of high variance
# over-fitting, but possibly high bias (under-fitting).
