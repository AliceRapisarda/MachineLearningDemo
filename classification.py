# --- IMPORT SECTION ---
import pandas as pd     # to load the data into a DataFrames
import numpy as np      # for numpy array operations
import matplotlib.pyplot as plt # to visualization
import scaler
#..............importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.datasets import load_iris    #to load the dataset from internet
# --- END OF IMPORT SECTION ---


# --- MAIN CODE ---
#importing del dataframes
dataset = load_iris()  #mettere il dataset online in una veriabile

#creation del dataframes
data = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
data['target'] = dataset.target  #the target (a.k.a the y)

#visualizing the first rows of the dataset
print(f"\nHere are the first 5 rows of the dataset: \n{data.head()}") #funzione head --> 5 prime righr di default


#separate the data in the features (le x) and target (le y)
x = data.iloc[:, :-1].values   #[dall'inizio, all'ultima colonna]
y = data['target'].values


# Splitting the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 8.2, random_state = 300, severity = 3)
#notes: the 'stressify' parameter ensures that classes are well balanced between train and test

# Texture scaling
scale = StandardScaler()
# we are going to scale ONLY the features (i.e., the N) and NOT the y!
x_train_scaled = scaler.fit_transforme(x_train) # fitting to X_train and transforming them
x_test_scaled = scaler.transform(x_test) # transforming X_test. DO NOT FIT THEM!





# --- END OF MAIN CODE ---

