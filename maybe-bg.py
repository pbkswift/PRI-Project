import tkinter as tk
from tkinter import messagebox
import random

class EducationalApp:
    def __init__(self, master):
        self.master = master
        self.master.title("FocusBoost")
        self.master.geometry("600x800")
        self.master.config(bg="lightblue")

        self.subject_options = ["Math", "Science", "English"]
        self.questions = {
            "Math": [
                {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "correct_answer": "4", "explanation": "2 + 2 equals 4."},
                {"question": "What is 5 x 5?", "options": ["20", "25", "30"], "correct_answer": "25", "explanation": "5 x 5 equals 25."},
                {"question": "What is 10 - 3?", "options": ["7", "8", "9"], "correct_answer": "7", "explanation": "10 - 3 equals 7 because when we take away 3 from 10 we are left with 7."},
                {"question": "What is 3 squared?", "options": ["6", "9", "12"], "correct_answer": "9", "explanation": "3 squared equals 9."},
                {"question": "What is 9 divided by 3?", "options": ["2", "3", "4"], "correct_answer": "3", "explanation": "9 divided by 3 equals 3."},
                {"question": "What is 4 + 6?", "options": ["10", "11", "12"], "correct_answer": "10", "explanation": "4 + 6 equals 10."},
                {"question": "What is 7 x 3?", "options": ["18", "20", "21"], "correct_answer": "21", "explanation": "7 x 3 equals 21."},
                {"question": "What is 20 - 15?", "options": ["4", "5", "6"], "correct_answer": "5", "explanation": "20 - 15 equals 5."},
                {"question": "What is 10 x 2?", "options": ["18", "20", "30"], "correct_answer": "20", "explanation": "10 x 2 equals 20."},
                {"question": "What is 18 divided by 2?", "options": ["8", "9", "10"], "correct_answer": "9", "explanation": "18 divided by 2 equals 9."}
            ],
            "Science": [
                {"question": "What is the chemical symbol for water?", "options": ["H2O", "CO2", "O2"], "correct_answer": "H2O", "explanation": "The chemical symbol for water is H2O."},
                {"question": "What planet is known as the Red Planet?", "options": ["Mars", "Venus", "Jupiter"], "correct_answer": "Mars", "explanation": "Mars is known as the Red Planet."},
                {"question": "What is the center of an atom called?", "options": ["Nucleus", "Electron", "Proton"], "correct_answer": "Nucleus", "explanation": "The center of an atom is called the nucleus."},
                {"question": "What is the closest star to Earth?", "options": ["Alpha Centauri", "Proxima Centauri", "Sun"], "correct_answer": "Sun", "explanation": "The closest star to Earth is the Sun."},
                {"question": "What gas do plants breathe in?", "options": ["Oxygen", "Carbon Dioxide", "Nitrogen"], "correct_answer": "Carbon Dioxide", "explanation": "Plants breathe in Carbon Dioxide."},
                {"question": "What is the largest organ in the human body?", "options": ["Liver", "Heart", "Skin"], "correct_answer": "Skin", "explanation": "The largest organ in the human body is the skin."},
                {"question": "What is the process by which plants make food called?", "options": ["Photosynthesis", "Respiration", "Transpiration"], "correct_answer": "Photosynthesis", "explanation": "The process by which plants make food is called Photosynthesis."},
                {"question": "What is the process by which water changes into vapor called?", "options": ["Evaporation", "Condensation", "Precipitation"], "correct_answer": "Evaporation", "explanation": "The process by which water changes into vapor is called Evaporation."},
                {"question": "What type of animal is a frog?", "options": ["Amphibian", "Reptile", "Mammal"], "correct_answer": "Amphibian", "explanation": "A frog is an Amphibian."},
                {"question": "What is the study of living organisms called?", "options": ["Biology", "Geology", "Physics"], "correct_answer": "Biology", "explanation": "The study of living organisms is called Biology."}
            ],
            "English": [
                {"question": "What is the opposite of 'happy'?", "options": ["Sad", "Angry", "Excited"], "correct_answer": "Sad", "explanation": "The opposite of 'happy' is 'sad'."},
                {"question": "What is the past tense of 'eat'?", "options": ["Ate", "Eaten", "Eating"], "correct_answer": "Ate", "explanation": "The past tense of 'eat' is 'ate'."},
                {"question": "Which word is a synonym for 'big'?", "options": ["Small", "Large", "Tiny"], "correct_answer": "Large", "explanation": "'Large' is a synonym for 'big'."},
                {"question": "What is the plural form of 'child'?", "options": ["Childs", "Childen", "Children"], "correct_answer": "Children", "explanation": "The plural form of 'child' is 'children'."},
                {"question": "What is the present participle of 'run'?", "options": ["Ran", "Running", "Runs"], "correct_answer": "Running", "explanation": "The present participle of 'run' is 'running'."},
                {"question": "What is a group of cows called?", "options": ["Flock", "Herd", "Pack"], "correct_answer": "Herd", "explanation": "A group of cows is called a 'herd'."},
                {"question": "Which word means 'to make something clean'?", "options": ["Dirty", "Wash", "Messy"], "correct_answer": "Wash", "explanation": "'Wash' means 'to make something clean'."},
                {"question": "What is the opposite of 'dark'?", "options": ["Bright", "Black", "Shadow"], "correct_answer": "Bright", "explanation": "The opposite of 'dark' is 'bright'."},
                {"question": "What is a word that has the opposite meaning of another word called?", "options": ["Synonym", "Homonym", "Antonym"], "correct_answer": "Antonym", "explanation": "A word that has the opposite meaning of another word is called an 'antonym'."},
                {"question": "What is the action of saying something to persuade, encourage, or reassure called?", "options": ["Compliment", "Praise", "Encourage"], "correct_answer": "Encourage", "explanation": "The action of saying something to persuade, encourage, or reassure is called 'encourage'."}
            ]
        }

        self.current_question = tk.StringVar()
        self.selected_subject = tk.StringVar()
        self.score = 0
        self.question_index = 0

        self.setup_ui()

    def setup_ui(self):
        # Title
        self.title_label = tk.Label(self.master, text="FocusBoost: ADHD Learning Buddy", font=("Arial", 24, "bold"), bg="lightblue")
        self.title_label.pack(pady=20)

        # Subject selection
        self.subject_label = tk.Label(self.master, text="Choose a subject:", font=("Arial", 16), bg="lightblue")
        self.subject_label.pack(pady=10)

        for subject in self.subject_options:
            button = tk.Button(self.master, text=subject, command=lambda subj=subject: self.start_quiz(subj), font=("Arial", 14), bg="orange", width=10)
            button.pack(pady=5)

        # Question display
        self.question_label = tk.Label(self.master, textvariable=self.current_question, font=("Arial", 20), bg="lightblue")
        self.question_label.pack(pady=10)

    def start_quiz(self, subject):
        self.selected_subject.set(subject)
        self.score = 0
        self.question_index = 0
        self.load_question()
        self.set_background(subject)

    def load_question(self):
        subject = self.selected_subject.get()
        question_data = self.questions[subject][self.question_index]
        self.current_question.set(question_data['question'])
        random.shuffle(question_data['options'])

        # Clear previous option buttons
        for widget in self.master.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()

        # Create new option buttons
        for option in question_data['options']:
            button = tk.Button(self.master, text=option, command=lambda opt=option: self.check_answer(opt), font=("Arial", 14, "bold"), bg="lightgray", width=20)
            button.pack(pady=5)

    def check_answer(self, selected_option):
        subject = self.selected_subject.get()
        question_data = self.questions[subject][self.question_index]
        correct_answer = question_data['correct_answer']
        explanation = question_data['explanation']
        if selected_option == correct_answer:
            self.score += 1
            messagebox.showinfo("Correct!", f"Your answer '{selected_option}' is correct!\nExplanation: {explanation}")
        else:
            messagebox.showinfo("Incorrect!", f"Your answer '{selected_option}' is incorrect.\nThe correct answer is '{correct_answer}'.\nExplanation: {explanation}")
        self.question_index += 1
        if self.question_index < len(self.questions[subject]):
            self.load_question()
        else:
            messagebox.showinfo("Quiz Over", f"Quiz for {subject} is over!\nYour Score: {self.score}/{len(self.questions[subject])}")

    def set_background(self, subject):
        if subject == "Math":
            self.master.config(bg="lightblue")
            bg_image = tk.PhotoImage(file="math_background.png")  # Path to your math background image
            bg_label = tk.Label(self.master, image=bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            bg_label.image = bg_image
        elif subject == "Science":
            self.master.config(bg="lightblue")
            bg_image = tk.PhotoImage(file="science_background.png")  # Path to your science background image
            bg_label = tk.Label(self.master, image=bg_image)
            bg_label.place(x=0, y=0, relwidth=1, relheight=1)
            bg_label.image = bg_image
        elif subject == "English":
            self.master.config(bg="lightblue")
            bg_image = tk.PhotoImage(file="/home/dk/Downloads/engwish.jpg")  # Path to your english background image
            bg_label = tk.Label(self.master, image=bg_image)
            bg_label.place(x=0, y=0, relwidth=100, relheight=100)
           # bg_label.image = bg_image

if __name__ == "__main__":
    root = tk.Tk()
    app = EducationalApp(root)
    root.mainloop()


