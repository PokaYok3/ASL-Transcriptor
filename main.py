import subprocess
import tkinter as tk
from tkinter import font
from time import strftime

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transcription application")
        self.geometry("800x600")
        self.configure(bg="#FFFF00")
        self.iconbitmap("images/logo.png")
        self.resizable(False, False)

        button_font = font.Font(family="Arial", size=12, weight="bold")

        self.clock_label = tk.Label(self, font=('calibri', 20, 'bold'), background='black', foreground='white')
        self.clock_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

        self.button_getdatatrain = tk.Button(self, text="Ejecutar getdatatrain.py", bg="#4CAF50", fg="white", font=button_font, command=self.execute_getdata_train)
        self.button_getdatatrain.place(relx=0.5, rely=0.2, anchor=tk.CENTER, width=200, height=40)

        self.button_getdatatest = tk.Button(self, text="Ejecutar getdatatest.py", bg="#9C27B0", fg="white", font=button_font, command=self.execute_getdatatest)
        self.button_getdatatest.place(relx=0.5, rely=0.3, anchor=tk.CENTER, width=200, height=40)

        self.button_trainmodel = tk.Button(self, text="Ejecutar trainmodel.py", bg="#FF9800", fg="white", font=button_font, command=self.execute_trainmodel)
        self.button_trainmodel.place(relx=0.5, rely=0.4, anchor=tk.CENTER, width=200, height=40)

        self.button_testmodel = tk.Button(self, text="Ejecutar testmodel.py", bg="#2196F3", fg="white", font=button_font, command=self.execute_testmodel)
        self.button_testmodel.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=200, height=40)

        self.button_transcription = tk.Button(self, text="Ejecutar transcription.py", bg="#F44336", fg="white", font=button_font, command=self.execute_transcription)
        self.button_transcription.place(relx=0.5, rely=0.6, anchor=tk.CENTER, width=200, height=40)

        self.update_clock()  # Llamar a la función para actualizar el reloj

    def update_clock(self):
        time_string = strftime('%H:%M:%S %p')
        self.clock_label.config(text=time_string)
        self.after(1000, self.update_clock)  # Actualizar cada 1000 milisegundos (1 segundo)

    def execute_getdata_train(self):
        subprocess.call(["python", "Scripts/getdatatrain.py"])

    def execute_getdatatest(self):
        subprocess.call(["python", "Scripts/getdatatest.py"])

    def execute_trainmodel(self):
        subprocess.call(["python", "Scripts/trainmodel.py"])

    def execute_testmodel(self):
        subprocess.call(["python", "Scripts/testmodel.py"])

    def execute_transcription(self):
        subprocess.call(["python", "Scripts/transcription.py"])

if __name__ == "__main__":
    window = MainWindow()
    window.mainloop()
