# --- IMPORT SECTION ---
import pandas as pd # for DataFrames
import numpy as np # for numpy array operations
import matplotlib.pyplot as plt # for visualization
#import seaborn as sns
#..............importing all the needed functions from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
# --- END OF IMPORT SECTION ---



# --- MAIN CODE ---

#importing del dataframes
path_to_data = "Salary_Data.csv"
data = pd.read_csv(path_to_data)
#print(data.head())

#visualizing del dataframes
print(f"\nHere are the first 5 rows of the dataset: \n{data.head()}")

#separate the data in features (può anche essere più di una colonna) and target
x = data['YearsExperience']
y = data['Salary']

#using a plot to visualize the data
plt.title("Years of Experience vs Salary") # title of the plot          (--> nome del grafico)
plt.xlabel("Years of Experience") # title of x axis                     (--> valori messi dell'asse x)
plt.ylabel("Salary") # title of y axis                                  (--> valori messi dell'asse y)
plt.scatter(data["YearsExperience"], data["Salary"], color="red") # actual plot
#..............sns.regplot(data = data, x = "YearsExperience", y = "Salary" ) #regression line      #(--> per disegnare una retta che rappresenta al meglio il grafico)
plt.show() # renderize the plot to show it

#splitting the dataset into training (porzione di addestramento) and test (por<ione di test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 101)
#significato:
#   test_size =  porzione del sataset che userà la parte di test (0.2 = 20% del totale)(quindi l'80% è usato per la parte di addestramento;
#   random_state = richiama diversi algoritmi del programma (101 = sarà un determinato algoritmo di estrazione casuale)


# --- END OF MAIN CODE ---