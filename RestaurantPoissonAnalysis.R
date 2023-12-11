library(tidyverse)
library(tidymodels)
library(poissonreg)
library(vroom)
library(glmnet)

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

## Trying Poisson Model
pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

pois_workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(pois_mod) %>%
  fit(data = train) # Fit the workflow

predictions_pois <- predict(pois_workflow,
                                 new_data=test) # Use fit to predict

## Get Predictions for test set AND format for Kaggle
test_preds_pois <- predict(pois_workflow, new_data = test) 

## Write prediction file to CSV for pois
vroom_write(x=test_preds_pois, file="poisson.csv", delim=",")
