# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from sklearn import metrics
from sklearn.externals import joblib
%matplotlib inline


# Importing the dataset
dataset = pd.read_csv('kc_house_data.csv')

#check number of rows & columns
dataset.shape

dataset.head(5)

dataset.tail(10)

#let us check if there are any nulls
dataset.isnull().any()

#dataset.info()

#check statistics
dataset.describe()


#visualize dataset
#scatterplot
sns.scatterplot(x = 'sqft_living', y = 'price', data=dataset)

#histogram
dataset.hist(bins = 20, figsize = (20,20), color='blue')

#check the correlation of features
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(dataset.corr(), annot= True)

sns.pairplot(dataset)

dataset_subcolumns = dataset[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']]

sns.pairplot(dataset_subcolumns)

#as there are 21 features, we will start with small features

selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

X = dataset[selected_features] #dataset.iloc[:, :-1].values

X.shape

y = dataset['price']
y.shape

#scale the features to be used in ann.... -1 to 1

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#verify scaled data

X_scaled.shape

scaler.data_max_
scaler.data_min_

y = y.values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)


#split to train & test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25, random_state = 0)

X_train.shape

#build model
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(50, input_dim = 7, activation = 'relu')) #50 neurons, 7 inputs, relu activation

model.add(Dense(50, activation= 'relu'))

model.add(Dense(1, activation = 'linear'))

#check out model summary
model.summary()

model.compile(optimizer = 'Adam', loss = 'mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2) #feeding data as 50, validation split is for cross valication & avoid overfitting


#evaluate model
epochs_hist.history.keys()


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('model loss progress during training')
plt.xlabel("Epoch")
plt.ylabel("Training and validation loss")
plt.legend(["Training loss", "validation loss"])


#dataset_subcolumns = dataset[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']]
x_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])
scaler_1 = MinMaxScaler()
x_test_scaled_1 = scaler_1.fit_transform(x_test_1)

y_predict_1 = model.predict(x_test_scaled_1)
y_predict_1 = scaler.inverse_transform(y_predict_1)


#full predict on test data
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color="blue")

y_predict_orig = scaler.inverse_transform(y_predict)
y_test_orig = scaler.inverse_transform(y_test)


k= X_test.shape[1]
n=len(X_test)


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)), '.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE = ' , RMSE , ' MSE= ' , MSE , ' MAE= ', MAE, ' R2=', r2, ' adjusted r2=', adj_r2)

#compare with linear regression
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred1 = regressor.predict(X_test1)

# let us get rsquare and coefficients
print('r_sq :', regressor.score(X, y))
print('co_ef_b0 ', regressor.intercept_)
co_ef_b1 = regressor.coef_


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test1, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test1, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred1)))
