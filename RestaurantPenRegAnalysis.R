library(vroom)
library(recipes)
library(glmnet)
library(parsnip)
library(workflows)

## Read in the data
train <- vroom("train.csv")
test <- vroom("test.csv")

## Create recipe
recipe <- recipe(revenue~., data=train) %>%
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

## Apply the recipe to data
prep <- prep(recipe)
baked <- bake(prep, new_data = train)

## Penalized lin regression model
model <- linear_reg(penalty=1, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set up workflow
wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(model) %>%
  fit(data=train)

## Predict
predictions <- predict(wf, new_data=test)

## Write penalized to CSV
vroom_write(x=predictions, file="./PenalizedRegression.csv", delim=",")
