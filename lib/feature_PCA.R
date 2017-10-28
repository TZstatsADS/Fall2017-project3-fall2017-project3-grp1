#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Grp 9 
### Project 3
### ADS Fall 2017

feature <- function(img_dir, n_dig = 0, n_pixel_row = 20, n_pixel_col = 20, desired_variance = 0.9, n_hogs = 54,
                    run.pca = T, run.hogs = T, export = T){
  
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
  #img0 <- resize(img0, n_pixel_row, n_pixel_col)
  
  mat1 <- as.matrix(img0)
  n_c <- nrow(mat1)
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, n_c) 
  
  ### run HOG feature extraction
  if(run.hogs){
    
    H <- matrix(NA, n_files, n_hogs)
    
    for(i in 1:n_files){
      
      img <- readImage( paste0( img_dir, list.files(img_dir)[i] , sep = "" ) )
      #img <- resize(img, n_pixel_row, n_pixel_col)

      h     <- HOG(img)
      H[i,] <- h
      
    }
    
  }
  
  if(run.pca){
    Hpca <- img.pca(H, desired_variance)
  }
  
  ### output constructed features
  if(export){
    
    save(H, file=paste0("../output/HOG.RData"))
    save(Hpca, file=paste0("../output/HOG-PCA.RData"))
    
  }
  
  return( list( "HOG" = H, "HPCA" = Hpca ) )
  
}


## perform PCA
img.pca <- function (dat, desired_variance = 0.9){  
  
  dat.pca <- prcomp(dat)
  
  variance.explained <- cumsum(dat.pca$sdev^2)/sum(dat.pca$sdev^2)
  n_pca <- match(1, variance.explained > desired_variance)
  
  print(paste("Number of principal components:", n_pca))
  
  dat <- dat.pca$x[,1:n_pca]
  return(dat)
  
}
