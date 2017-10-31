#########################################################
### Train a classification model with training images ###
#########################################################

### Authors: Grp 1 
### Project 3
### ADS Fall 2017

train <- function( dat_train, label_train, params=NULL,
                   run.gbm = F, run.svm = F, run.rf = F,
                   run.lda = F, run.qda = F){
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ###  -  boolean variables indicating which models to run
  ###  -  'params' model parameters
  ### Output: training model specification
  
  ### Train with gradient boosting model
  
  
  gbm <- NULL
  if( run.gbm ){
    gbm <- gbmFit( dat_train, label_train, params )
    return( gbm )
  }
  
  
  ### Train random forest model
  randomForest <- NULL
  if( run.rf ){
    #randomForest <- randomForestFit( dat_train, label_train, par )
    #return( randomForest )
    mtry = as.integer(params[1])
    ntree = as.integer(params[2])
    #dat = cbind(label_train[,2],dat_train)
    if(!require("randomForest")){
      install.packages("randomForest")
    }
    library(randomForest)
    rf.model <- randomForest(as.factor(label_train[,2]) ~ .,
                             data = dat_train, mtry = mtry,
                             importance=TRUE, 
                             ntree=ntree)
    return(rf.model)
  }
  
  
  ### Train SVM model
  svm <- NULL
  if( run.svm ){
    svm <- svmFit( dat_train, label_train, params )
    return( svm )
  }
  
  ### train LDA model
  lda <- NULL
  if( run.lda ){
    lda <- ldaFit( dat_train, label_train )
    return( lda )
  }
    
}

## GBM model
gbmFit <- function( dat_train, label_train, params){
  library("gbm")
  if(!require("gbm")){
    install.packages("gbm")
  }
  library("gbm")
  fitGbm <- gbm.fit(x = dat_train, y=label_train[,2],
                     n.trees = 300,
                     shrinkage = params,
                     distribution = "multinomial",
                     interaction.depth = 1, 
                     bag.fraction = 0.5,
                     verbose = FALSE,
                     n.minobsinnode = 1)
  
  best_ntrees <- gbm.perf(fitGbm, method="OOB", plot.it = FALSE)
  
  return( list( fit = fitGbm, best_n.trees=best_ntrees) )
  
}


## SVM model
svmFit <- function( dat_train, label_train, params ){ 
  
  if(!require("e1071")){
    install.packages("e1071")
  }
  
  library("e1071")
  
  svm.model <- svm(dat_train, label_train[,2], type = "C", kernel = "radial", gamma = params)
  
  return(list(fit = svm.model))
}

## CNN model
cnnFit <- function( ){ # fill with necessary paramsameters
  #para need: direction of files, thus no paras here
  experiment_dir <- "../data/training_set/"
  img_train_dir <- paste(experiment_dir, "train/", sep="")
  label_dir <- "../data/training_set/label_train.csv"
  
  
}

## QDA fit
qdaFit <- function( dat_train, label_train ){
  
  fit <- qda(x = dat_train$HPCA, grouping = label_train[,2], method = "mle", CV = T)
  return( list( fit  = fit ) )

}

#LDA fit
ldaFit <- function( dat_train, label_train ){
  
  fit <- lda(x = dat_train, grouping = label_train[,2], CV = T )
  return( list( fit  = fit ) )
  
}



