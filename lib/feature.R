#############################################################
### Construct visual features for training/testing images ###
#############################################################

### Authors: Christina Huang yh2859
### Project 3
### ADS Fall 2017

feature <- function(img_dir, export=T){
  
  ### Construct process features for training/testing images
  ### Feature: HOG Values for each photos
  
  ### Input: a directory that contains images ready for processing
  ### Output: a matrix contains processed features for the images

  library("EBImage")
  library("OpenImageR")
  
  #img_dir <- "../data/zipcode/test"
  
  n_files <- length(list.files(img_dir))
  dir_names <- list.files(img_dir) # determine img dimensions
  
  H <- matrix(NA, n_files, 54) # save HOG Values, each row indicating each photo

  ### store vectorized pixel values of images
  dat <- matrix(NA, n_files, n_r) 
  for(i in 1:n_files){
    img <- readImage(paste(img_dir, dir_names[i], sep = "/"))
    h <- HOG(img)
    H[i,] <- h
  }
  
  ### output constructed features
  if(export){
    save(H, file=paste0("../output/HOG.RData"))
  }
  return(H)
}




