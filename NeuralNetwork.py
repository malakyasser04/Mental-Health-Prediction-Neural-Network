#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

sleep_duration_mapping = {
    'Less than 5 hours': 1,
    '5-6 hours': 2,
    '6-7 hours': 3,
    '6-8 hours': 4,
    '7-8 hours': 5,
    '8-9 hours': 6,
    '9-11 hours': 7,
    '10-11 hours': 8,
    'More than 8 hours': 9
}

dietary_habits_mapping = {
    'Healthy': 2,
    'Moderate': 1,
    'Unhealthy': 0
}

df['Dietary Habits'] = df['Dietary Habits'].replace({
    'More Healthy': 'Healthy',
    'Less Healthy': 'Unhealthy',
    'Less than Healthy': 'Moderate',
    'No Healthy': 'Unhealthy'
})

df['Sleep Duration'] = df['Sleep Duration'].replace({
    '1-2 hours': 'Less than 5 hours',
    '4-6 hours':  'Less than 5 hours'    ,
    '2-3 hours': 'Less than 5 hours' ,
    '3-4 hours':  'Less than 5 hours',
    '4-5 hours': 'Less than 5 hours' ,
    '1-3 hours':  'Less than 5 hours',
    '1-6 hours':  'Less than 5 hours' ,
    'than 5 hours': 'Less than 5 hours' ,
    '8 hours':'7-8 hours' ,
    '3-6 hours':'Less than 5 hours'   ,
})
valid_sleep_duration=['Less than 5 hours','5-6 hours' ,'6-7 hours' ,'6-8 hours' ,'7-8 hours' ,'8-9 hours' ,'9-11 hours' ,'10-11 hours','More than 8 hours']
valid_dietary_habits = ['Healthy', 'Moderate', 'Unhealthy']
df = df[df['Dietary Habits'].isin(valid_dietary_habits)]
df['Dietary Habits'] = df['Dietary Habits'].map(dietary_habits_mapping)

df = df[df['Sleep Duration'].isin(valid_sleep_duration)]
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_duration_mapping)


df.drop(['id', 'Name', 'Gender', 'City', 'Working Professional or Student',
         'Profession', 'CGPA', 'Degree'], axis='columns', inplace=True, errors='ignore')

encoder = LabelEncoder()
for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
    df[col] = encoder.fit_transform(df[col])

df.fillna(0, inplace=True)

target = df['Depression']
features = df.drop('Depression', axis=1)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)





# In[24]:


#first model
model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),  
    layers.Dense(128, activation='sigmoid'),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train,epochs=25,batch_size=100,validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# In[2]:


model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),  
    layers.Dense(128, activation='sigmoid'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='sigmoid'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train,epochs=25,batch_size=100,validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# In[12]:


from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),  
     layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(0.01)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01)),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train,epochs=25,batch_size=100,validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# In[15]:


from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential

model = Sequential([
    layers.Input(shape=(X_train.shape[1],)),  
     layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(0.01)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01)),
    layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
, loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train,epochs=25,batch_size=100,validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



# In[16]:


import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.optimizers import Adam


model = Sequential([
    layers.Input(shape=(X_train.shape[1],)), 
    layers.Dense(128, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(64, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=25, batch_size=100, validation_split=0.2)


test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[27]:


new_data = pd.read_csv('test.csv')


new_data['Sleep Duration'] = new_data['Sleep Duration'].replace({
    '1-2 hours': 'Less than 5 hours',
    '4-6 hours':  'Less than 5 hours'    ,
    '2-3 hours': 'Less than 5 hours' ,
    '3-4 hours':  'Less than 5 hours',
    '4-5 hours': 'Less than 5 hours' ,
    '1-3 hours':  'Less than 5 hours',
    '1-6 hours':  'Less than 5 hours' ,
    'than 5 hours': 'Less than 5 hours' ,
    '8 hours':'7-8 hours' ,
    '3-6 hours':'Less than 5 hours'   ,
})
new_data['Dietary Habits'] = new_data['Dietary Habits'].replace({
    'More Healthy': 'Healthy',
    'Less Healthy': 'Unhealthy',
    'Less than Healthy': 'Moderate',
    'No Healthy': 'Unhealthy'
})


# new_data.loc[~new_data['Dietary Habits'].isin(valid_dietary_habits), 'Dietary Habits'] = 0
# new_data.loc[~new_data['Sleep Duration'].isin(valid_sleep_duration), 'Sleep Duration'] = 0

# new_data['Dietary Habits'] = new_data['Dietary Habits'].map(dietary_habits_mapping).fillna(0)
# new_data['Sleep Duration'] = new_data['Sleep Duration'].map(sleep_duration_mapping).fillna(0)

new_data = new_data[new_data['Dietary Habits'].isin(valid_dietary_habits)]
new_data['Dietary Habits'] = new_data['Dietary Habits'].map(dietary_habits_mapping)


new_data = new_data[new_data['Sleep Duration'].isin(valid_sleep_duration)]
new_data['Sleep Duration'] = new_data['Sleep Duration'].map(sleep_duration_mapping)

ids=new_data['id']

new_data.drop(['id', 'Name', 'Gender', 'City', 'Working Professional or Student',
               'Profession', 'CGPA', 'Degree'], axis='columns', inplace=True, errors='ignore')



for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
    new_data[col] = encoder.transform(new_data[col])

new_data.fillna(0, inplace=True)

test_data_scaled = scaler.transform(new_data)

predictions = model.predict(test_data_scaled)
binary_predictions = (predictions > 0.5).astype(int)

print("Binary Predictions:",binary_predictions)

# submission_df = pd.DataFrame({
#     'id': ids, 
#     'Depression': binary_predictions.flatten()  
# })

# submission_df.to_csv('C:/Users/malak/Downloads/submission.csv', index=False)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# In[4]:


def init_params():
    input_size = X_train.shape[1]
    hidden_size = input_size * 2
    output_size = 1
    W1=np.random.rand(hidden_size, input_size)-0.5
    b1=np.random.rand(hidden_size, 1)-0.5
    W2=np.random.rand(output_size, hidden_size)-0.5
    b2=np.random.rand(output_size, 1)-0.5
    return W1, b1, W2, b2

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_deriv(Z):
    return sigmoid(Z)*(1-sigmoid(Z))

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z>0

def forward_prop(W1, b1, W2, b2, X):
    Z1=W1.dot(X.T)+b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m=X.shape[0]  
    dZ2=(A2-Y.T)*sigmoid_deriv(Z2)
    dW2=1/m*dZ2.dot(A1.T)
    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1= W2.T.dot(dZ2)*ReLU_deriv(Z1)
    dW1=1/m*dZ1.dot(X)
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1=W1-alpha*dW1
    b1=b1-alpha*db1
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    return W1, b1, W2, b2

def compute_cost(A2,Y):
    m=Y.shape[0]
    return -1/m*np.sum(Y*np.log(A2.T)+(1-Y)*np.log(1-A2.T))

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        cost = compute_cost(A2, Y)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss Cost: {cost}")
    return W1, b1, W2, b2


alpha = 0.5
iterations = 2000
y_train_reshaped = y_train.values.reshape(-1, 1)
W1, b1, W2, b2 = gradient_descent(X_train, y_train_reshaped, alpha, iterations)

def get_predictions(A2):
    return (A2>0.5).astype(int)

def predict(X,W1,b1,W2,b2):
    _, _, _,A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

predictions = predict(X_test, W1, b1, W2, b2)
accuracy = np.mean(predictions.flatten() == y_test.values) * 100

print(f"Test Accuracy: {accuracy:.2f}%")


# In[ ]:




