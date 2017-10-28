#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Yuting Ma/Tian Zheng
### Project 3
### ADS Spring 2017

feature <- function(img_dir, n_dig = 4, n_pixel_row = 20, n_pixel_col = 20, export = T){
  
  ### Construct process features for training/testing images
  ### Sample simple feature: Extract row average raw pixel values as features
  
  ### Input: a directory that contains images ready for processing
  ### Output: an .RData file contains processed features for the images
  
  ### load libraries
  library("EBImage")
  
  n_files <- length(list.files(img_dir)[-1])
  
  ### determine img dimensions
  img_dir <- paste( img_dir, "/", sep = "")
  img0 <-  readImage(paste0(img_dir, "img_", do.call(paste, c(as.list(rep("0",(n_dig-1))), sep = "")), 1, ".jpg"))
  #img0 <-  readImage( paste( img_dir, "/img_0001.jpg", sep = "" ) )
  img0 <- resize(img0, n_pixel_row, n_pixel_col)
  mat1 <- as.matrix(img0)
  n_r <- nrow(img0)
  
  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, n_r) 
  for(i in 1:n_files){
    img <- readImage(paste0(img_dir,  "img_", do.call(paste, c(as.list(rep("0",(n_dig-nchar(i)))), sep = "")), i, ".jpg"))
    img <- resize(img, n_pixel_row, n_pixel_col)
    dat[i,] <- rowMeans(img)
  }
  
  ### output constructed features
  if(export){
    save(dat, file=paste0("../output/feature_", ".RData"))
  }
  return(dat)
}
