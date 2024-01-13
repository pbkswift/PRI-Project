import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

data = pd.read_csv(
    "./dataset/Training.csv")#.dropna(axis = 1)

print("Training.csv-----------------")
print(data)

print ("Training.csv minus last column with no values ----")

data = data.dropna(axis=1)

print(data)

print ("Prognosis Column Only -----------")

print(data["prognosis"])

print ("Magik of Encoding -----")

encoder = LabelEncoder()
encodedarray = encoder.fit_transform(data["prognosis"])

print(encodedarray)

print("Replacing the prognosis Column")

data["prognosis"] = encodedarray
print(data)

print("Get Table X via iLoc'ing --INPUTS for MACHINE LEARNING---")

X = data.iloc[:,:-1]
print(X)

print("Get Table Y via iLoc'ing --CORRESPONDING OUTPUT for MACHINE LEARNING---")

y = data.iloc[:, -1]
print(y)

print("START TRAINING ----------------")

# Support Vector Classifier
model = SVC(C=1, probability=True)
model.fit(X.values, y)

print("TRAINING COMPLETE ----------------")

print ("GET SYMPTOMS COLUMN INDEX ------------")

symptoms = X.columns.values
print (symptoms)

def find_column_index_of(symptom):
    for index, value in enumerate(symptoms):
        if (value == symptom):
            return index

print(find_column_index_of("itching"))
print(find_column_index_of("skin_rash"))
print(find_column_index_of("nodal_skin_eruptions"))
print(find_column_index_of("yellow_crust_ooze"))

print("CREATE INPUT FOR PREDICTION ----------------")

input_data = [0]*len(X.columns)

input_data[find_column_index_of("itching")] = 1
input_data[find_column_index_of("skin_rash")] = 1
input_data[find_column_index_of("nodal_skin_eruptions")] = 1
input_data[find_column_index_of("yellow_crust_ooze")] = 1

print (input_data)

print ("RESHAPING INPUT ------")

reshaped_input_data = np.array(input_data).reshape(1, -1)
print(reshaped_input_data)

print ("PREDICT ------")

prediction_prob = model.predict_proba(reshaped_input_data)
prediction = model.predict(reshaped_input_data)

print ("Prediction is ")
print(prediction)

print ("Probability is ")
print(prediction_prob)

print ("Which in human language is -----")

print (encoder.classes_[prediction[0]])
print ("with a probability of ",
       format(prediction_prob[0][prediction[0]], ".0%"))
