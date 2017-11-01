#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Grp 1 
### Project 3
### ADS Fall 2017

feature <- function(img_dir, n_dig = 0, n_pixel_row = 20, n_pixel_col = 20, 
                    desired_variance = 0.9, n_hogs = 54, run.pca = F, 
                    run.hogs = F, run.cnn = F, run.lbp = F, export = T){
  
  ### Construct process features for training/testing images
  ### Sample simple feature: Extract row average raw pixel values as features
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains processed features for the images
  
  ### load libraries
  library("EBImage")
  library("jpeg")
  library("OpenImageR")

  n_files <- length(list.files(img_dir))
  
  ### determine img dimensions
  img0 <- readImage( paste0( img_dir, list.files(img_dir)[1] , sep = "" ) )
  img0 <- resize(img0, n_pixel_row, n_pixel_col)
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, nrow(as.matrix(img0) ) ) 
  
  ### run HOG feature extraction
  H <- NULL
  if(run.hogs){
    
    H <- matrix(NA, n_files, n_hogs)
    
    for(i in 1:n_files){
      
      img <- readImage( paste0( img_dir, list.files(img_dir)[i] , sep = "" ) )
      img <- resize(img, n_pixel_row, n_pixel_col)

      h     <- HOG(img)
      H[i,] <- h
      
    }
    
    save(H, file=paste0("../output/hogFeatures.RData"))
    return( H )
    
  }
  
  ## run PCA on HOG features
  pcaFeatures <- NULL
  if(run.pca){
    pcaFeatures <- pcaFeatureExtraction(H, desired_variance)
    save(pcaFeatures, file=paste0("../output/pcaHogFeatures.RData"))
    return( pcaFeatures)
  }
  
  ## run CNN feature extraction
  cnnFeatures <- NULL
  if( run.cnn ){
    cnnFeatures <- cnnFeatureExtraction(img_dir)
    save(cnnFeatures, file=paste0("../output/cnnFeatures.RData"))
    return( cnnFeatures )
  }
  
  ## run LBP feature extraction
  lbpFeatures <- NULL
  if( run.lbp ){
    lbpFeatures <- lbpFeatureExtraction()
    save(lbpFeatures, file=paste0("../output/lbpFeatures.RData"))
    return( lbpFeatures )
  }
  
#  return( list( "HOG" = H, "HogPca" = pcaFeatures, 
#                "CNN" = cnnFeatures, "LBP" = lbpFeatures ) )
  
}


## perform PCA
pcaFeatureExtraction <- function (dat, desired_variance = 0.9){  
  
  dat.pca <- prcomp(dat)
  
  variance.explained <- cumsum(dat.pca$sdev^2)/sum(dat.pca$sdev^2)
  n_pca <- match(1, variance.explained > desired_variance)
  
  print(paste("Number of principal components:", n_pca))
  
  dat <- dat.pca$x[,1:n_pca]
  return(dat)
  
}


## perform CNN - insert necessary parameters 
cnnFeatureExtraction <- function(img_dir){ 
  #img_dir = "../data/training_set/train/"
  cmd <- paste("~/anaconda/bin/python ../lib/CNN_test.py", img_dir)
  system(cmd)

  dat <- read.csv("../output/feature_CNN_test.csv")
  dat <- dat[,-1]
  return(dat)
}


## perform LBP - insert necessary parameters
lbpFeatureExtraction <- function(){
  
}
