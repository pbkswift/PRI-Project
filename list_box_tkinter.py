import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class DiagnoseApp:
    def __init__(self, root):
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
            font=font_DiagnosisTitle)
        self.diag2_title = ttk.Label(self.diag2_frame, text="NB Prediction:",
            font=font_DiagnosisTitle)
        self.diag3_title = ttk.Label(self.diag3_frame, text="SVM Prediction:",
            font=font_DiagnosisTitle)
        
        self.diag1_diagnosis = ttk.Label(self.diag1_frame, text="..",
            font=font_DiagnosisTitle)
        self.diag2_diagnosis = ttk.Label(self.diag2_frame, text="..",
            font=font_DiagnosisTitle)
        self.diag3_diagnosis = ttk.Label(self.diag3_frame, text="..",
            font=font_DiagnosisTitle)

        self.diag1_title.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag2_title.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag3_title.pack(side=TOP, fill=X, padx=10, pady=5)

        self.diag1_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag2_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)
        self.diag3_diagnosis.pack(side=TOP, fill=X, padx=10, pady=5)

        self.center_mid_section.config(width=50)
        self.dest_list.config(width=33)

        items = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
        for item in items:
            self.source_list.insert(tk.END, item)


        self.root.bind("<Configure>", self.on_resize)

    def search(self):
        search_term = self.search_entry.get()
        print("Searching for:", search_term)

    def add_item(self):
        selected_indexes = self.source_list.curselection()
        selected_symptoms = [self.source_list.get(index) for index in selected_indexes]
        for item in selected_symptoms:
            self.dest_list.insert(tk.END, item)

        for index in selected_indexes[::-1]:
            self.source_list.delete(index)

    def diagnose_button_clicked(self):
        pass

    def remove_item(self):
        selected_indexes = self.dest_list.curselection()
        selected_symptoms = [self.dest_list.get(index) for index in selected_indexes]
        for item in selected_symptoms:
            self.source_list.insert(tk.END, item)

        for index in selected_indexes[::-1]:
            self.dest_list.delete(index)

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
