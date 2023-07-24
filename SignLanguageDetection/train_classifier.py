import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./data.pickle', 'rb'))

#print(data_dict.keys())
#print(data_dict)

# data and labels are list so need to convert them into numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# split data into train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
# Always shuffle your data!
# Make stratify=labels ! It allows to split all labels proportionally for train and test data

# Model
model = RandomForestClassifier()

# Train
model.fit(x_train, y_train)

# Test
y_predict = model.predict(x_test)

# Performance evaluation
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score*100))

# Save the model - can use this model later on
f = open('model.pickle', 'wb')  # 'wb' --> writing, bytes
pickle.dump({'model': model}, f)
f.close()  # close the file

