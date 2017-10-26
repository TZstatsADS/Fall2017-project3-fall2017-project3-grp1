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
  
  img.pca <- function (dat, desired_variance = 0.9){  
    
    dat.pca <- prcomp(dat)
    
    variance.explained <- cumsum(dat.pca$sdev^2)/sum(dat.pca$sdev^2)
    n_pca <- match(1, variance.explained > desired_variance)
    
    print(paste("Number of principal components:", n_pca))
    
    dat <- dat.pca$x[,1:n_pca]
    return(dat)
  }
  
  n_files <- length(list.files(img_dir))
  
  ### determine img dimensions
  img0 <-  readImage(paste0(img_dir, "img_", do.call(paste, c(as.list(rep("0",(n_dig-1))), sep = "")), 1, ".jpg"))
  img0 <- resize(img0, n_pixel_row, n_pixel_col)
  
  
  mat1 <- as.matrix(img0)
  n_c <- nrow(mat1)
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, n_c) 
  if(run.hogs){H <- matrix(NA, n_files, n_hogs)}
  
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir,  "img_", do.call(paste, c(as.list(rep("0",(n_dig-nchar(i)))), sep = "")), i, ".jpg"))
    img <- resize(img, n_pixel_row, n_pixel_col)
    dat[i,] <- img
    
    if(run.hogs){
    h <- HOG(img)
    H[i,] <- h
    }
  }
  
  
  if(run.pca){
    dat.pca <- img.pca(dat, desired_variance)
  }
  
  ### output constructed features
  if(export){
    
    if(run.pca){save(dat.pca, file=paste0("../output/feature_PCA.RData"))}
    
    if(run.hogs){save(H, file=paste0("../output/HOG.RData"))}
    
    }
  
  dat <- cbind(dat.pca, H)
  
  return(dat)
}
