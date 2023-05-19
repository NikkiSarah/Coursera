library(tidymodels)

data("iris")

set.seed(101)

data_split <- initial_split(iris, prob = 0.2, strata = Species)
train_data <- training(data_split)
test_data <- testing(data_split)

# define the recipe
knn_rec <- recipe(Species ~., data = train_data) %>% step_normalize()

# define the model
knn_spec <- nearest_neighbor(neighbors = 2) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

# define the workflow
iris_wflow <- workflow() %>% 
  add_model(knn_spec) %>% 
  add_recipe(knn_rec)
iris_wflow

# fit the model
knn_fit <- iris_wflow %>% fit(data = train_data)
knn_fit

# assess performance
knn_aug <- augment(knn_fit, test_data)
metrics(knn_aug, Species, .pred_class)

