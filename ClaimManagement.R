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


