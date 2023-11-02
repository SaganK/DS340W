import numpy as np
import pandas as pd  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load data from an Excel file 
data = pd.read_excel('Cyprus_nM=50,dtp=15.xlsx')


## Data Processing Steps

# Shuffle Data
shuffled_data = data.sample(frac=1, random_state=57) 

# Reset the index to avoid any issues with the shuffled data
shuffled_data.reset_index(drop=True, inplace=True)

data = shuffled_data


column = 'Actual Mag'  
bin_edges = [0, 3.0, 5.0, 6.0, 10.0]  

bin_labels = ['Small', 'Medium', 'Large', 'Catastrophic']

data['Binned_Column'] = pd.cut(data[column], bins=bin_edges, labels=bin_labels)


# Split data into features (X) and labels (y)
X = data.iloc[:, :60].values  
y = data.iloc[:, 60].values  

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=51)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=51)

# Standardize 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create ANN Model
model = Sequential()
model.add(Dense(64, input_dim=60, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['MSE'])

# Train the model
out = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=150, batch_size=32)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)