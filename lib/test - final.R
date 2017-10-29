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
  
  #fit_train = fit 
  #dat_test = test.data
  #test.rf = T
  
  #print('hello')
  ## test LDA
  if( test.lda ){
    pred <- predict(fit_train$fit, newdata = dat_test )

    pred <- pred$class
  }
  
  ## test SVM
  if( test.svm ){
    pred <- predict(fit_train$fit, newdata = dat_test )
  }
  
  ## test GBM
  if( test.gbm ){
    
   prob.pred <- predict(fit_train$fit, newdata=dat_test, 
                    n.trees=fit_train$best_n.trees, type="response")
    pred<- apply(prob.pred, 1, which.max) 
    pred<- pred-1
     
 } 
  
  if( test.rf ){
    #print(dim(dat_test))
    pred <- predict(fit_train, newdata = dat_test)
  }
  
  return(pred)
  
}

