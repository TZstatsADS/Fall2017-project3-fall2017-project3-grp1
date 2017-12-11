# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2017

+ Team 1
+ Team members
	+ Christina Huang
	+ Henrique Saboya
	+ Yi Zhang
	+ Joaquim Lyrio
	+ Jing Zhao

+ Project summary: In this project, we proposed an improvement on the current classification engine to classify dogs, muffins and fried chicken in terms of running cost (storage, memory and time) and prediction accuracy. The baseline model for comparison uses Gradient Boosting Machine (GBM) with decision stumps on 5000 SIFT features. Our proposed model consists of a Convolutional Neural Network with 2 convolutional layers. Before attaining the best model, we tried a different range of feature extraction methods, such as HOG and PCA, as well as different classification models, like LDA, Random Forest and SVM. We can observe the prediction accuracy estimated by cross-validation of each model using CNN feature extraction in the following table:

![image](figs/Table.jpg)
	
**Contribution statement**: 

Christina Huang - Established and optimized Convolution Neural Network(CNN/Python) model; worked on HOG and two versions (3/128 dimension) of CNN feature extraction; used flipping and rotation to expand dataset for 200% and significantly improve accuracy of CNN model for 40%; implemented all connection of Python code and .Rmd files.

Henrique Saboya - Tested different models such as SVM, LDA, QDA; worked on HOG and PCA feature extraction, tried other different image segmentation as feature extractions, such as color matrix norms.

Jing Zhao - Establied and optimized Random Forest(RF) model, used ggplot for visualizing its results; worked on LBP feature extraction and tried other classification models like logistics model; combined above models with .Rmd files.

Joaquim Lyrio - Created the structure of the project's code, such as the train, test and feature files; tested different models such as SVM, LDA, QDA; worked on HOG and PCA feature extraction; worked on the baseline model structure and tried to find more efficient ways to run it; wrote great part of the main.Rmd file

Yi Zhang - Baseline model and Presentation

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
