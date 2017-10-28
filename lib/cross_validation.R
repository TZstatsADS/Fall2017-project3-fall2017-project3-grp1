########################
### Cross Validation ###
########################

### Authors: Grp 1 
### Project 3
### ADS Fall 2017


cv.function <- function(X.train, y.train, d, K,
                        cv.svm = F, cv.gbm = F, cv.rf = F,
                        cv.lda = F){
  
  # to debug
  #X.train <- dat_train
  #y.train <- label_train
  #d = 3
  #cv.lda = T

  
  n <- length(y.train[,2])
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i,]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i,]
    
    ## cross validate to GBM model
    if( cv.gbm ){
      
      par <- list(depth=d)
      fit <- train(train.data, train.label, par, run.gbm = TRUE)
      
      pred <- test(fit, test.data, test.gbm = T)
      
    }
    
    ## cross validate to SVM model
    if( cv.svm ){
      par <- list(gamma=d)
      fit <- train(train.data, train.label, par, run.svm = TRUE)
      
      fit <- fit$svm
      
      pred <- test(fit, test.data, test.svm = T)
      
    }
    
    ## cross validate to RF model
    if( cv.rf ){
      #par <- list(gamma=d) # CHANGE THIS
      fit <- train(train.data, train.label, par, run.rf = T)
      fit <- fit$rf
      
      pred <- test(fit, test.data, test.rf = T)
    }
    
    ## cross validate LDA model
    if( cv.lda ){
      
      fit <- train(train.data, train.label, par = NULL, run.lda = T)
      
      fit <- fit$fit
      
      pred <- test( fit, test.data, test.lda = T )
      
    }
    
    cv.error[i] <- mean(pred != y.train[s == i,2])  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}
