library(tidyverse)
library(tidymodels)
library(vroom)

# Read in the data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Create recipe
recipe <- recipe(revenue ~ ., data = train) %>% 
  step_normalize(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_novel(all_nominal_predictors())

# Apply the recipe to data
prep <- prep(recipe)
baked <- bake(prep, new_data = train)

# Random Forest
my_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = tune()  # Consider expanding the search range
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(my_mod)

# Set up grid of tuning values 
tuning_grid <- grid_regular(
  mtry(range = c(1, 10)),  # Adjust the range for mtry
  min_n(range = c(1, 50)),  # Adjust the range for min_n
  trees(range = c(500, 1000)),  # Adjust the range for trees
  levels = 4
) 

# Set up K-fold CV
folds <- vfold_cv(train, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- forest_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(rmse),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")

## Finalize the Workflow & fit it
final_forest_wf <-
  forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# Make predictions on the test set
forest_predictions <- predict(final_forest_wf, new_data = test)

# Save predictions to a file
forest_predictions <- cbind(test$Id, forest_predictions)
vroom_write(x = forest_predictions, file = "forest.csv", delim = ",")

## SVM model
svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

# Create a workflow with model & recipe
svm_wf <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(svmLinear)

# Set up grid of tuning values 
tuning_grid <- grid_regular(cost(), levels = 3) 

# Set up K-fold CV
folds <- vfold_cv(amazonTrain, v = 5, repeats = 1)

# Find best tuning parameters 
CV_results <- svm_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc),
    control = control_grid(verbose = TRUE)  # Enable verbose output to monitor progress
  )

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

## Finalize the Workflow & fit it
final_svm_wf <-
  svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
final_svm_wf %>%
  predict(new_data = amazonTrain, type="prob")

svm_predictions <- predict(final_svm_wf,
                             new_data=amazonTest,
                             type="prob") # "class" or "prob" (see doc)

svm_predictions <- cbind(amazonTest$id, svm_predictions) %>%
  rename(Id = "amazonTest$id",
         Action = ".pred_1") #%>%
#select (-"amazonTest$id")

vroom_write(x=svm_predictions, file="svm.csv", delim=",")


