# interactive-prediction-ui.py

import pandas as pd
import tkinter as tk

def diagnose_clicked():
    selected_indexes = listbox.curselection()
    for selected_index in selected_indexes:
        print(listbox.get(selected_index))

def setup_gui(symptoms):
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

    for symptom in symptoms:
        listbox.insert(
            tk.END,
            symptom
        )

    button = tk.Button(
        root,
        text="Diagnose",
        command=diagnose_clicked
    )
    button.place(x=420, y=20, width=100, height=20)

    return listbox

data = pd.read_csv(
    "./dataset/Training.csv").dropna(axis = 1)
X = data.iloc[:,:-1]
symptoms = X.columns.values

root = tk.Tk()
root.title = "Diagnose ..."
root.geometry('600x400+50+50')
listbox = setup_gui(symptoms)

root.mainloop()
