######################################################
### Fit the classification model with testing data ###
######################################################

### Authors: Grp 1 
### Project 3
### ADS Fall 2017

test <- function(fit_train, dat_test, 
                 test.lda = F, test.gbm = F, test.svm = F, test.rf = F){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  - processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ## test LDA
  if( test.lda ){
    pred <- predict(fit_train, newdata = dat_test )
  }
  
  ## test GBM
  if( test.gbm ){
    pred <- predict(fit_train$fit, newdata=dat_test, 
                    n.trees=fit_train$iter, type="response") 
  }
  
  return(as.numeric(pred> 0.5))
  
}

