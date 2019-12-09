
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Churn_customers.csv")

x= dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_1.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:];


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))


model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))

model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))


model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 10, epochs = 10)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


prediction = model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
prediction = (prediction > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_improved_model():
    model = Sequential()
    model.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))
    model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
    model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    model.compile(optimizer ='adam', loss='binary_crossentropy', metrics = ['accuracy'])
    return model

new_model = KerasClassifier(build_fn = build_improved_model, batch_size = 10, nb_epoch = 50)
accuracies = cross_val_score(estimator = new_model, X = x_train, y = y_train, scoring='accuracy', cv = 10)

mean = accuracies.mean()
variance =accuracies.std()


#improving ann
#dropout regualization to reduce overfitting if needed
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_improved_modelCV(optimizer):
    model = Sequential()
    model.add(Dense(activation = 'relu', input_dim = 11, units = 6, kernel_initializer = 'uniform'))
    model.add(Dense(activation = 'relu', units = 6, kernel_initializer = 'uniform'))
    model.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    return model

new_model_cv = KerasClassifier(build_fn = build_improved_modelCV)

parameters = {'batch_size' : [25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = new_model_cv, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X = x_train, y = y_train)

best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_




