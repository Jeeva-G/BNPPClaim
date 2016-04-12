#Load the libraries
library(xgboost)

# load training, testing, and sample_submission data sets..
train <- read.table("../inputdata_github/train.csv", sep = ",", header = TRUE)
test <- read.table("../inputdata_github/test.csv", sep = ",", header = TRUE)
sample_submission <- read.table("../inputdata_github/sample_submission.csv", sep = ",", header = TRUE)