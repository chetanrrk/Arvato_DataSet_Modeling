# Arvato_DataSet_Modeling
This work was done as a part of Udacity machine learning engineering capstone project. The dataset are available as a part of kaggle compitition at
https://www.kaggle.com/c/udacity-arvato-identify-customers/overview

Datasets:
•	Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).

•	Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).

•	Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).

•	Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

# Modeling Approach

# 1) Data cleaning and processing steps

  a. Identifying NaN entries and remove features with >30% NaN. This could be played around with to improve accuracy.
  
  b. Converting categorical features into numerical using binary encoding for binary features, one-hot-encoding for features with less category, and numerical encoding for larger categories where memory requirement for on-hot encoding becomes impractical. This procedure can also be more fine tuend for improving accuracy.
  
  c. Imputing the NaNs using most frequent occurance. Since the data is categorical, this approach is a better choice. 
  
  d. Scaling the data using minmax scaler in sklearn and also trying standard scaler in sklearn
  
# 2) Preparing the data for 10-fold cross validation preparation

  a. The dataset was highly imbalanced. Undersampling the overly represented class was undertaken. 
  
  b. Some amount of the training data was left behind for crossvalidation from both classes. 10% of the sparse data and 30% of majority class was kept in the validation set. 

# 3) Metrics of success

  a. average accuray: The accuracy is average of the correct predictions is obtained between the truth (y_i) and the prediction ((y_i ) ̂). 
  
  b. Area under the ROC curve (AUC): For the binary classification this is identical to the average of the sensitivity and specificity
  AUC=  1/2 (TP/(TP+FN)+ TN/(TN+FP))
 
# 4) Algorithms used
 
  a. Minibathc KMeans: The population segmentation was done using the Sklearns version of minibatch KMeans was applied to the population data. Nealy a 900,000 records and high dimensionality prevented the direct use of KMeans. Minibatch allows to process a bath randomly sampled at a time and is known to give similar performance to the KMeans algorithm.
  
  b. Linear Descriminant Analysis (LDA): Sklearn version of LDA was tried on the dataset and checked for various undersampling ratio of the two classes. The best model was chosen based on validation data during the 10-fold cross validation procedure

  c. XGBoost Algorith: XGBoost container available for the AWS sagemaker instance was used for this process. Because of the calss imbalance, grid search for the 'scale_pos_weight' parameter was performed on the training and selection based on validation data.
  
  d. Best performing model was obtained from the validation process and applied to the test data. The model was deployed on AWS for generating labels on the test data.
 
 # 5) Codes
 
  a. clean_population_datasets.ipynb: basic exploration and data cleaning for demographic data set 'Udacity_AZDIAS_052018.csv'
  
  b. clean_customer_dataset.ipynb: basic exploration and data cleaning for demographic data set 'Udacity_CUSTOMERS_052018.csv'
  
  c. clean_test_dataset.ipynb: basic exploration and data cleaning for demographic data set 'Udacity_MAILOUT_052018_TEST.csv'
  
  c. clustering_population.ipynb: performs minibatch KMeans clustering on the population data and projects the customer data onto the population data.
  
  d. lda_model.ipynb: performs cleaning and transformation of the training data and applies LDA algorithm to build the model.
  
  e. xgboost_model.ipynb: performs cleaning and transformation of the training data and applies XGBoost algorithm to build the model.
  
