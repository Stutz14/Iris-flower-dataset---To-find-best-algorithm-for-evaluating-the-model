# Iris-Project

A machine learning project is certainly not linear and involves numerous steps which if not implemented carefully, could result in a bad model.
The steps required to successfully build an accurate model are as follows:

1. Define the problem clearly.
2. Prepare the data.
3. Evaluate various different kinds of algorithms.
4. Improve the results obtained.
5. Present the results.

**The dataset used**: Iris Flower Dataset (https://en.wikipedia.org/wiki/Iris_flower_data_set). 
There are four columns of measurements of the flowers in centimeters. The fifth column is the species of the flower observed. All observed flowers 
belong to one of three species.

To begin with, I played around with the data and obtained numerous discrete summaries (both numerical and 
graphical). It helped me gain insight of the data at a high level. Summarizing the data proves to be useful in the long run since it gives an 
overview of the dataset and what to expect.

There isn't a particular problem to be specified in this project. This project basically showcases how different algorithms can be used on a dataset 
and how the best algorithm can be found out and used to evaluate the model on unseen data using python.

The data was cleaned up before many algorithms were tried on it. Technically, this step is called data pre-processing. I cleaned the data by removing 
any outliers which could affect the data in a huge way and scaled each entry by making mean = 0 and variance = 1 of the dataset. The dataset was then 
divided into training dataset which comprises about 80% of the transformed dataset and validation dataset which comprises about 20% of the 
transformed dataset.

Several algorithms were used in order to successfully compute the best model. The algorithms used are:
  1. Logistic Regression (LR)
  2. Linear Discriminant Analysis (LDA)
  3. K-Nearest Neighbors (KNN).
  4. Classification and Regression Trees (CART).
  5. Gaussian Naive Bayes (NB).
  6. Support Vector Machines (SVM)

The best algorithm was selected after careful consideration and the best one was used to evaluate the model on the validation dataset and make 
predictions.
