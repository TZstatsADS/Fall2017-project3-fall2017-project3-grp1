#############################################################
### Construct visual features for training/testing images
### function "feature_HOG"
#############################################################

### Authors: Christina Huang yh2859
### Project 3
### ADS Fall 2017

feature_HOG <- function(img_dir, export=T){
  
  ### Construct process features for training/testing images
  ### Feature: HOG Values for each photos
  
  ### Input: a directory that contains images ready for processing
  ### Output: a matrix contains processed features for the images

  library("EBImage")
  library("OpenImageR")
  
  #img_dir <- "../data/training_set/train"
  
  n_files <- length(list.files(img_dir))
  dir_names <- list.files(img_dir) # determine img dimensions
  
  
  ### store vectorized pixel values of images
  H <- matrix(NA, n_files, 54) # save HOG Values, each row indicating each photo
  for(i in 1:n_files){
    img <- readImage(paste(img_dir, dir_names[i], sep = "/"))
    h <- HOG(img)
    H[i,] <- h
    if(i %% 10 == 0) print(paste(i, "pictures already been featured"))
  }
  
  ### output constructed features
  if(export){
    save(H, file=paste0("../output/HOG_feature.RData"))
  }
  return(H)
}

write.csv(H, "../data/feature_HOG.csv")


