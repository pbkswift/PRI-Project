# interactive-prediction-ui.py

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tkinter as tk

## Function Definitions

# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

def train_model(model_name, model, x, y, x_test, y_test):
    model.fit(x, y)

    print(f"Accuracy on train data by {model_name}\
       : {accuracy_score(y, model.predict(x))*100}" )
    
    # preds = model.predict(x_test)
    # cf_matrix = confusion_matrix(y_test, preds)

    print(f"Accuracy on train data by {model_name}\
       : {accuracy_score(y_test, model.predict(x_test))*100}" )

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):
    # symptoms = symptoms.split(",")
    
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
        
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
    
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # making final prediction by taking mode of all predictions
    # final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    final_prediction = np.unique([rf_prediction, nb_prediction, svm_prediction])[0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions

def diagnose_button_clicked():
    selected_indexes = listbox.curselection()
    if len(selected_indexes)>2:
        selected_symptoms = [listbox.get(index) for index in selected_indexes]
        predictions = predictDisease(selected_symptoms)
        label_rf_prediction.config(text=predictions["rf_model_prediction"])
        label_nb_prediction.config(text=predictions["naive_bayes_prediction"])
        label_svm_prediction.config(text=predictions["svm_model_prediction"])
    else:
        print("Select at least 3 symptoms")
        label_rf_prediction.config(text="Select at least 3 symptoms")
        label_nb_prediction.config(text="Select at least 3 symptoms")
        label_svm_prediction.config(text="Select at least 3 symptoms")

# Starts here
# Pandas has a function called read_csv
# Dropna is to remove missing values
# Reading the train.csv by removing the last column since it's an empty column
data = pd.read_csv("./dataset/Training.csv").dropna(axis = 1)

# Reading the test data
test_data = pd.read_csv("./dataset/Testing.csv").dropna(axis=1)

# Encoding the target value into numerical value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

X = data.iloc[:,:-1]
y = data.iloc[:, -1]

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

train_model("SVM Classifier", final_svm_model, X, y, test_X, test_Y)
train_model("Naive Bayes Classifier", final_nb_model, X, y, test_X, test_Y)
train_model("Random Forest Classifier", final_rf_model, X, y, test_X, test_Y)

# Making prediction by take mode of predictions 
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

final_preds = [mode([i,j,k])[0] for i,j,k in zip(svm_preds, nb_preds, rf_preds)]

print(f"Accuracy on Test dataset by the combined model\
: {accuracy_score(test_Y, final_preds)*100}")

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

root = tk.Tk()
root.title = "Diagnose ..."
root.geometry('800x400+50+50')

listbox = tk.Listbox(
    root,
    selectmode=tk.MULTIPLE
)
listbox.place(height=400, width=400)

scroll_bar = tk.Scrollbar(root, orient='vertical',
    command=listbox.yview)
scroll_bar.place(x=400, y=0, width=10, height=400)

#  communicate back to the scrollbar
listbox['yscrollcommand'] = scroll_bar.set

thekeys = data_dict["symptom_index"].keys()
thekeys = list(thekeys)
thekeys = sorted(thekeys)

for symptom in thekeys:
    listbox.insert(
        tk.END,
        symptom
    )

button = tk.Button(
    root,
    text="Diagnose",
    command=diagnose_button_clicked
)
button.place(x=420, y=20, width=250, height=20)

label1 = tk.Label(
    root,
    text = "RF Model Prediction:",
    anchor=tk.W
)
label1.place(x=420, y=45, width=250, height=20)

label2 = tk.Label(
    root,
    text = "Naive Bayes Prediction:",
    anchor=tk.W
)
label2.place(x=420, y=95, width=250, height=20)

label3 = tk.Label(
    root,
    text = "Support Vector Machine Prediction:",
    anchor=tk.W
)
label3.place(x=420, y=145, width=250, height=20)

label_rf_prediction= tk.Label(
    root,
    text = "RF",
    anchor=tk.W
)
label_rf_prediction.place(x=440, y=70, width=250, height=20)
label_nb_prediction= tk.Label(
    root,
    text = "NB",
    anchor=tk.W
)
label_nb_prediction.place(x=440, y=120, width=250, height=20)
label_svm_prediction= tk.Label(
    root,
    text = "SVM",
    anchor=tk.W
)
label_svm_prediction.place(x=440, y=170, width=250, height=20)

root.mainloop()
