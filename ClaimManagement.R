############ Load the libraries #########

library(xgboost)

############ load training, testing, and sample_submission data sets.. #########

train <- read.table("../inputdata_github/train.csv", sep = ",", header = TRUE)
test <- read.table("../inputdata_github/test.csv", sep = ",", header = TRUE)
sample_submission <- read.table("../inputdata_github/sample_submission.csv", sep = ",", header = TRUE)

############  Pre-processing ##########

# Remove ID Column
train <- train[, -1]
test <- test[, -1]

# Set all NAs to -1
train[is.na(train)] <- -1
test[is.na(test)] <- -1

# Store names of variables for future
train_names <- names(train[, -1])

#Converting the dataset to DMatrix format as the xgboost model requires it.
#training data has target column and so removing it.
train <- xgb.DMatrix(data = data.matrix(train[, -1]), label = data.matrix(train$target))

#test data does not have target column, so directly using it.
test <- xgb.DMatrix(data = data.matrix(test))

########## Model building/training using training data set ##########

#Set seed to allow reproducibility of the benchmark value
set.seed(2016)

#Model building
model1 <- xgboost(data = train, max.depth = 5, eta = 0.5, nround = 30, eval.metric = "logloss", objective = "binary:logistic")

######### Model Analysis #########

# Finding important variables from the xgboost trained model
importance_model <- xgb.importance(model = model1, feature_names = train_names)

# Print the top 20 best picked variables submitted to xgboost
print(importance_model, nrows = 20)

######### Predict data on testing set ########

# predict values of the trained model on test data
predictedValues <- predict(model1, newdata = test)

######### prediction: creating submission file ########

# Applies the probabilities found previously from the prediction to the submission sample
sample_submission$PredictedProb <- predictedValues

# export to CSV the sample submission for submitting to Kaggle
write.csv(sample_submission, file = "../inputdata_github/sample_submission1.csv", row.names = FALSE)