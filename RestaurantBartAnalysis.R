library(tidyverse)
library(tidymodels)
library(vroom)
library(dbarts)

# Read in the data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Create recipe
recipe <- recipe(revenue ~ ., data = train) %>%
  step_rename(Open_date = `Open Date`,
              City_group = `City Group`) %>%
  step_mutate(Open_date = as.Date(Open_date, format = "%m/%d/%Y"),
              City = factor(City),
              City_group = factor(City_group),
              Type = factor(Type),
              Id = factor(Id)) %>%
  step_normalize(all_numeric_predictors()) %>%
  update_role(Id, new_role = "ID") %>%
  step_date(Open_date, features = c("dow", "month", "year", "quarter")) %>%
  step_rm(Id, Open_date) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_novel(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_numeric_predictors())

# Apply the recipe to data
prep <- prep(recipe)
baked <- bake(prep, new_data = train)

##Bart
bart_mod <- parsnip::bart(mode = "regression",
                          engine = "dbarts", 
                          trees = 20)

bart_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(bart_mod) 

# Fit the model on your training data
bart_fit <- fit(bart_wf, data = train)

# Now you can make predictions
bart_predictions <- predict(bart_fit, new_data = test)

## Write prediction file to CSV for tree tuning
vroom_write(x = bart_predictions, file = "bart.csv", delim = ",")
