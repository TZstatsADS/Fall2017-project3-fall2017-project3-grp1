#########################################################
### Train a classification model with training images ###
#########################################################

### Authors: Grp 1 
### Project 3
### ADS Fall 2017

train <- function( dat_train, label_train, par=NULL,
                   run.gbm = F, run.svm = F, run.rf = F,
                   run.lda = F, run.qda = F){
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ###  -  boolean variables indicating which models to run
  ###  -  par parameter for gbm
  ### Output: training model specification
  
  ### Train with gradient boosting model
  gbm <- NULL
  if( run.gbm ){
    gbm <- gbmFit( dat_train, label_train, par )
    return( gbm )
  }
  
  
  ### Train random forest model
  randomForest <- NULL
  if( run.rf ){
    randomForest <- randomForestFit( dat_train, label_train )
    return( randomForest )
  }
  
  
  ### Train SVM model
  svm <- NULL
  if( run.svm ){
    svm <- svmFit( dat_train, label_train, par )
    return( svm )
  }
  
  ### train LDA model
  lda <- NULL
  if( run.lda ){
    lda <- ldaFit( dat_train, label_train )
    return( lda )
  }
  
  
  # Return fitted models
  #return( list( gbm = gbm, rf = randomForest, svm = svm, lda = lda ) )
  
}

## GBM model
gbmFit <- function( dat_train, label_train, par ){
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  
  ## fit model
  fitGbm <- gbm.fit(x = dat_train, y=label_train[,2],
                     n.trees = 20,
                     distribution = "multinomial",
                     interaction.depth = depth, 
                     bag.fraction = 0.5,
                     verbose = FALSE)
  
  best_iter <- gbm.perf(fitGbm, method="OOB", plot.it = FALSE)
  
  return( list( fit  = fitGbm, iter = best_iter ) )
  
}

## Ranfom Forest model
randomForestFit <- function( dat_train, label_train ){ # fill with necessary parameters
  
}

## SVM model
svmFit <- function( dat_train, label_train, par ){ 
  
  if(!require("e1071")){
    install.packages("e1071")
  }
  
  library("e1071")
  
  svm.model <- svm(dat_train, label_train[,2], type = "C", kernel = "radial", gamma = par)
  
  return(list(fit = svm.model))
}

## CNN model
cnnFit <- function( dat_train, label_train ){ # fill with necessary parameters
  
}

## QDA fit
qdaFit <- function( dat_train, label_train ){
  
  fit <- qda(x = dat_train$HPCA, grouping = label_train[,2], method = "mle", CV = T)
  return( list( fit  = fit ) )

}

#LDA fit
ldaFit <- function( dat_train, label_train ){
  
  fit <- lda(x = dat_train, grouping = label_train[,2] )
  return( list( fit  = fit ) )
  
}



