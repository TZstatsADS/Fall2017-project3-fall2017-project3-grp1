#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Yuting Ma
### Project 3
### ADS Spring 2016


train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  library(data.table)
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  
  ## comment this later
  dat_train   <- fread("/Users/pedrohmariani/Documents/Columbia/Academic/Fall 2017/AppliedDataScience/Projects/TZstatsFolder/Fall2017-project3-fall2017-project3-grp1/data/training_set/sift_train.csv", 
                       sep = ",")
  label_train <- read.csv("/Users/pedrohmariani/Documents/Columbia/Academic/Fall 2017/AppliedDataScience/Projects/TZstatsFolder/Fall2017-project3-fall2017-project3-grp1/data/training_set/label_train.csv")
  fit_gbm <- gbm.fit(x=dat_train[,2:ncol(dat_train)], y=label_train[,2],
                     n.trees = 50,
                     distribution="multinomial",
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)

  return(list(fit=fit_gbm, iter=best_iter))
}
