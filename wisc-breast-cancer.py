import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing, svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from yellowbrick.regressor import ResidualsPlot


# part 0 - loading the dataset
wisc = pd.read_excel("breast-cancer-wisconsin.xlsx") # loading the data

# part 1 - divide the data up into training and testing data
X = pd.DataFrame(np.c_[wisc["Clump Thickness"], wisc["Cell Size Uniformity"], wisc["Cell Shape Uniformity"], 
                      wisc["Marginal Adhesion"], wisc["Single Epithelial Cell Size"], wisc["Bland Chromatin"], 
                      wisc["Normal Nucleoli"], wisc["Mitoses"]]) # load all feature vectors
Y = wisc["Class"] # load the value to predict

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30) # divide the dataset into training and testing

# part 2 - train and test the model and visualize the results
model = SVC() # load the model we want to use, SVM
model.fit(X_train, y_train) # train the model using the training dataset
predictions = model.predict(X_test) # use the now trained model to predict on the testing dataset

print(classification_report(y_test, predictions)) # visualize the results

wisc_mal = wisc