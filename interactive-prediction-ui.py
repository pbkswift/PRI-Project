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
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class DiagnoseApp:
    def __init__(self, root):
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

        data_X = data.iloc[:,:-1]
        data_y = data.iloc[:, -1]

        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])

        # Training the models on whole data
        self.__final_svm_model = SVC(C=1, probability=True)
        self.__final_nb_model = GaussianNB()
        self.__final_rf_model = RandomForestClassifier(random_state=18)

        self.train_model("SVM Classifier", self.__final_svm_model, data_X, data_y, test_X, test_Y)
        self.train_model("Naive Bayes Classifier", self.__final_nb_model, data_X, data_y, test_X, test_Y)
        self.train_model("Random Forest Classifier", self.__final_rf_model, data_X, data_y, test_X, test_Y)

        # Making prediction by take mode of predictions 
        # made by all the classifiers
        svm_preds = self.__final_svm_model.predict(test_X)
        nb_preds = self.__final_nb_model.predict(test_X)
        rf_preds = self.__final_rf_model.predict(test_X)

        final_preds = [mode([i,j,k])[0] for i,j,k in zip(svm_preds, nb_preds, rf_preds)]

        print(f"Accuracy on Test dataset by the combined model\
        : {accuracy_score(test_Y, final_preds)*100}")

        symptoms = data_X.columns.values

        # Creating a symptom index dictionary to encode the
        # input symptoms into numerical form
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index

        self.__data_dict = {
            "symptom_index":symptom_index,
            "predictions_classes":encoder.classes_
        }

        thekeys = self.__data_dict["symptom_index"].keys()
        thekeys = list(thekeys)
        thekeys = sorted(thekeys)

        self.root = root
        self.root.title("Diagnose App")
        self.root.geometry('800x400+50+50')
        self.__knownWidth = 800

        font_FrameTitle = ("Arial", 12)
        font_DiagnosisTitle = ("Arial", 10, "bold")

        self.top_frame = ttk.Frame(self.root)
        self.bottom_frame = ttk.Frame(self.root)
        self.center_frame = ttk.Frame(self.root)

        self.top_frame.pack(side=TOP, fill=X)
        self.bottom_frame.pack(side=BOTTOM, fill=X)
        self.center_frame.pack(side=TOP, fill=BOTH, expand=TRUE)

        self.search_entry = ttk.Entry(self.top_frame,
                                     bootstyle="primary")
        self.search_button = ttk.Button(self.top_frame,
                                       text="Search", command=self.search,
                                        bootstyle="primary")
        self.search_entry.pack(side="left", fill="x",
                               expand=True, padx=10, pady=5)
        self.search_button.pack(side="left", padx=10, pady=5)

        self.center_frame_label = ttk.Label(self.center_frame,
            text="Select Symptoms:", font=font_FrameTitle)
        self.dest_list = tk.Listbox(self.center_frame,selectmode=tk.MULTIPLE)
        self.center_mid_section = ttk.Frame(self.center_frame)
        self.source_list = tk.Listbox(self.center_frame, selectmode=tk.MULTIPLE)

        self.center_frame_label.pack(side=TOP, fill=X, padx=10, pady=5)
        self.dest_list.pack(side="right", fill="y", padx=10, pady=5)
        self.center_mid_section.pack(side="right", fill="y")
        self.source_list.pack(side="right", fill="both", expand=TRUE, padx=10, pady=5)

        self.add_button = ttk.Button(self.center_mid_section,
            text=">>", command=self.add_item, bootstyle="primary")
        self.remove_button = ttk.Button(self.center_mid_section,
            text="<<", command=self.remove_item, bootstyle="secondary")
        self.diagnose_button = ttk.Button(self.center_mid_section,
            text="Diagnose", bootstyle=SUCCESS, command=self.diagnose_button_clicked)

        self.add_button.pack(side="top", padx=10, pady=5)
        self.remove_button.pack(side="top", padx=10, pady=5)
        self.diagnose_button.pack(side="bottom", padx=10, pady=5)

        self.bottom_frame_label = ttk.Label(self.bottom_frame,
            text="Diagnosis:", font=font_FrameTitle)
        self.diag1_frame = ttk.Frame(self.bottom_frame, bootstyle="primary")
        self.diag2_frame = ttk.Frame(self.bottom_frame, bootstyle="secondary")
        self.diag3_frame = ttk.Frame(self.bottom_frame, bootstyle="info")

        self.bottom_frame_label.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag1_frame.pack(side=LEFT, fill=Y, padx=10, pady=5)
        self.diag2_frame.pack(side=LEFT, fill=Y, padx=10, pady=5)
        self.diag3_frame.pack(side=LEFT, fill=Y, padx=10, pady=5)

        self.diag1_title = ttk.Label(self.diag1_frame, text="RF Model Prediction:",
            font=font_DiagnosisTitle, bootstyle="primary")
        self.diag2_title = ttk.Label(self.diag2_frame, text="NB Prediction:",
            font=font_DiagnosisTitle, bootstyle="secondary")
        self.diag3_title = ttk.Label(self.diag3_frame, text="SVM Prediction:",
            font=font_DiagnosisTitle, bootstyle="info")
        
        self.diag1_diagnosis = ttk.Label(self.diag1_frame, text="Select at least 3 symptoms",
            font=font_DiagnosisTitle, bootstyle="primary")
        self.diag2_diagnosis = ttk.Label(self.diag2_frame, text="Select at least 3 symptoms",
            font=font_DiagnosisTitle, bootstyle="secondary")
        self.diag3_diagnosis = ttk.Label(self.diag3_frame, text="Select at least 3 symptoms",
            font=font_DiagnosisTitle, bootstyle="info")

        self.diag1_title.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag2_title.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag3_title.pack(side=TOP, fill=X, padx=10, pady=5)

        self.diag1_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag2_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag3_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)

        self.center_mid_section.config(width=50)
        self.dest_list.config(width=33)

        self.set_symptoms(thekeys)

        self.root.bind("<Configure>", self.on_resize)

    # Defining scoring metric for k-fold cross validation
    def cv_scoring(self, estimator, X, y):
        return accuracy_score(y, estimator.predict(X))

    def train_model(self, model_name, model, x, y, x_test, y_test):
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
    def predictDisease(self, symptoms):
        # symptoms = symptoms.split(",")
        
        # creating input data for the models
        input_data = [0] * len(self.__data_dict["symptom_index"])
        for symptom in symptoms:
            index = self.__data_dict["symptom_index"][symptom]
            input_data[index] = 1
            
        # reshaping the input data and converting it
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
        
        # generating individual outputs
        rf_prediction_prob = self.__final_rf_model.predict_proba(input_data)
        print(rf_prediction_prob.max())
        nb_prediction_prob = self.__final_nb_model.predict_proba(input_data)
        svm_prediction_prob = self.__final_svm_model.predict_proba(input_data)

        rf_prediction = self.__data_dict["predictions_classes"][self.__final_rf_model.predict(input_data)[0]]
        nb_prediction = self.__data_dict["predictions_classes"][self.__final_nb_model.predict(input_data)[0]]
        svm_prediction = self.__data_dict["predictions_classes"][self.__final_svm_model.predict(input_data)[0]]
        
        # making final prediction by taking mode of all predictions
        # final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
        final_prediction = np.unique([rf_prediction, nb_prediction, svm_prediction])[0]
        predictions = {
            "rf_model_prediction": rf_prediction,
            "rf_model_prediction_prob": rf_prediction_prob.max() * 100, 
            "naive_bayes_prediction": nb_prediction,
            "naive_bayes_prediction_prob": nb_prediction_prob.max() * 100,
            "svm_model_prediction": svm_prediction,
            "svm_model_prediction_prob": svm_prediction_prob.max() * 100,
            "final_prediction":final_prediction
        }
        return predictions

    def add_item(self):
        selected_indexes = self.source_list.curselection()
        selected_symptoms = [self.source_list.get(index) for index in selected_indexes]
        for item in selected_symptoms:
            self.dest_list.insert(tk.END, item)

        for index in selected_indexes[::-1]:
            self.source_list.delete(index)

    def diagnose_button_clicked(self):
        if self.dest_list.size()>2:
            selected_symptoms = [self.dest_list.get(index) for index in range(self.dest_list.size())]
            predictions = self.predictDisease(selected_symptoms)
            self.diag1_diagnosis.config(text= predictions["rf_model_prediction"])
            self.diag2_diagnosis.config(text= predictions["naive_bayes_prediction"])
            self.diag3_diagnosis.config(text=predictions["svm_model_prediction"])
            # self.diag1_diagnosis.config(
            #     text= "{0} @ {1:1.2f}%".format(predictions["rf_model_prediction"], predictions["rf_model_prediction_prob"]))
            # self.diag2_diagnosis.config(
            #     text= "{0} @ {1:1.2f}%".format(predictions["naive_bayes_prediction"], predictions["naive_bayes_prediction_prob"]))
            # self.diag3_diagnosis.config(
            #     text="{0} @ {1:1.2f}%".format(predictions["svm_model_prediction"], predictions["svm_model_prediction_prob"]))
        else:
            print("Select at least 3 symptoms")
            self.diag1_diagnosis.config(text="Select at least 3 symptoms")
            self.diag2_diagnosis.config(text="Select at least 3 symptoms")
            self.diag3_diagnosis.config(text="Select at least 3 symptoms")

    def remove_item(self):
        selected_indexes = self.dest_list.curselection()
        selected_symptoms = [self.dest_list.get(index) for index in selected_indexes]
        for item in selected_symptoms:
            self.source_list.insert(tk.END, item)

        for index in selected_indexes[::-1]:
            self.dest_list.delete(index)

    def search(self):
        search_term = self.search_entry.get()
        print("Searching for:", search_term)

    def set_symptoms(self, symptoms):
        self.__symptoms = symptoms
        for symptom in symptoms:
            self.source_list.insert(
                tk.END, symptom)

    def on_resize(self, event):
        # Resize search entry and button along with the window
        if self.__knownWidth != event.width and self.root == event.widget:
            self.__knownWidth = event.width
            self.search_entry.config(width=event.width // 15)
            self.search_button.config(width=event.width // 50) 

def main():
    root = ttk.Window(themename="minty")
    app = DiagnoseApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()