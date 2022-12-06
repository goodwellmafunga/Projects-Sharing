#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # GENERAL SOLUTION

# In[2]:


concrete_data = pd.read_csv('C:/Users/SECURITY/Desktop/concrete.csv')

concrete_data.head()


# In[3]:


concrete_data.shape


# 
#  There are approximately 1000 samples to train our model on.To avoid overfitting, we have to be careful with our sample size

# In[4]:


concrete_data.describe()


# In[5]:


concrete_data.isnull().sum()


# The above indicates that our data is very clean and we therefore have to split our data into target variable and predictor variables.
# 

# In[7]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'strength']] # all columns except Strength
target = concrete_data['strength'] # Strength column


# In[8]:


predictors.head()


# In[9]:


target.head()


# In[10]:


num_columns = predictors.shape[1] # number of predictors
num_columns


# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# In[17]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(num_columns,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Model training and testing 

# In[18]:


# build the model
model = regression_model()


# # Question A 

# In[19]:



# fit the model on 50 epochs
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[20]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[21]:


from sklearn.metrics import mean_squared_error
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[22]:




total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# # Question B using normalised form

# In[23]:



predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=42)


# In[25]:


# fit the model on 50 epochs
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[26]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[27]:


from sklearn.metrics import mean_squared_error
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[28]:


epochs = 50
mean_squared_errors = []
for i in range(0, epochs):
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# we can see that before normalising  we got Mean: 37.26975988649361
# Standard Deviation: 10.075029546696246
#  and
# after normalising Mean: 23.549346955271595
# Standard Deviation: 8.464415123119538  which infer that data is less deviated  from the path after normalising.
# 

# # question C changing epoch to 100

# In[29]:


# fit the model on 100 epochs
epochs = 100
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[30]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[31]:


from sklearn.metrics import mean_squared_error
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[33]:


total_mean_squared_errors = 50
epochs = 100
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# deviation is less than B

# # QD adding 3 hidden layers with 10 neurons 

# In[37]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(num_columns,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[38]:


# build the model
model = regression_model()


# In[39]:


# fit the model on 50 epochs
epochs = 50
model.fit(X_train, y_train, epochs=epochs, verbose=1)


# In[40]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# In[41]:


from sklearn.metrics import mean_squared_error
mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# In[42]:


total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)

print('\n')
print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# In[ ]:




