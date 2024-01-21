import tkinter as tk

def print_selected_names():
    selected_names = listbox.curselection()
    names = [listbox.get(index) for index in selected_names]
    print("Selected Names:", names)

# Create the main window
root = tk.Tk()
root.title("List Box Example")

# Create a listbox and add three names
names = ["Shah Rukh Khan", "Tom Cruise", "Taylor Swift"]
listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
for name in names:
    listbox.insert(tk.END, name)
listbox.pack()

# Create a button to print selected names
button = tk.Button(root, text="Print Selected Names", command=print_selected_names)
button.pack()

# Run the application
root.mainloop()
