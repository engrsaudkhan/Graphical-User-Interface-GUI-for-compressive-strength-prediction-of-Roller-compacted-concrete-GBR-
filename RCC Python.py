import tkinter as tk
from tkinter import ttk
from math import pow, sqrt
from PIL import Image, ImageTk
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor 
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import BaggingRegressor 
from sklearn.metrics import mean_squared_error

class RangeInputGUI:
    def __init__(self, master):
        self.master = master
        master.title("GUI for Predicting Compressive Strength of Roller Compacted Concrete")
        master.configure(background="#FFFFFF")
        window_width = 650
        window_height = 730
        x_cord = 0  # Start from the left edge of the screen
        y_cord = 0  # Start from the top edge of the screen
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        x_cord = 0
        y_cord = 0
        master.geometry(f"{window_width}x{window_height}+{x_cord}+{y_cord}")
        main_heading = tk.Label(master, text="GUI for Predicting Compressive Strength of \n Roller Compacted Concrete",
                                bg="lightblue", fg="purple", font=("Helvetica", 16, "bold"), pady=10)
        main_heading.pack(side=tk.TOP, fill=tk.X)
        self.content_frame = tk.Frame(master, bg="#E8E8E8")
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=50, pady=50, anchor=tk.CENTER)
        self.canvas = tk.Canvas(self.content_frame, bg="#E8E8E8")
        self.scrollbar = ttk.Scrollbar(self.content_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#FFFFFF")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.input_frame.pack(side=tk.TOP, fill="both", padx=10, pady=10, expand=False)
        heading = tk.Label(self.input_frame, text="Input Parameters", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), padx=10, pady=10)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.output_frame = tk.Frame(self.scrollable_frame, bg="#FFFFFF", bd=2, relief=tk.RIDGE)
        self.output_frame.pack(side=tk.TOP, fill="both", padx=20, pady=20)
        heading = tk.Label(self.output_frame, text="Output Parameter", bg="#FFFFFF", fg="black", font=("Helvetica", 16, "bold"), pady=10)
        heading.grid(row=0, column=0, columnspan=2, pady=10)
        self.create_entry("W/C:", 0.4, 1)
        self.create_entry("Fly ash:", 0, 3)
        self.create_entry("Sand:", 1132.2, 5)
        self.create_entry("Gravel:", 667.8, 7)
        self.create_entry("Age:", 1, 9)
        heading.grid(row=0, column=0, columnspan=3, pady=10)
        self.gbr_button_GBR = tk.Button(self.output_frame, text="Gradient Boosting Regressor (GBR)", command=self.calculate_G_Bagging_regressor,
                                        bg="#743089", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.gbr_button_GBR.grid(row=3, column=0, pady=10, padx=10, sticky="sw")
        self.br_output_text_gradientbaggingregressor = tk.Text(self.output_frame, height=1.5, width=20)
        self.br_output_text_gradientbaggingregressor.grid(row=3, column=1, padx=10, pady=10)
        self.predict_button = tk.Button(self.output_frame, text="Predict", command=self.predict,
                                        bg="green", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.predict_button.grid(row=4, column=0, pady=10, padx=10)

        self.clear_button = tk.Button(self.output_frame, text="Clear", command=self.clear_fields,
                                      bg="red", fg="white", font=("Helvetica", 12, "bold"), relief=tk.RAISED)
        self.clear_button.grid(row=4, column=1, pady=10, padx=10)

        developer_info = tk.Label(text="This GUI is developed by: \n Muhammad Saud Khan (khans28@myumanitoba.ca), University of Manitoba, Canada\n",
                                  bg="#D0F0C0", fg="purple", font=("Helvetica", 11, "bold"), pady=10)
        developer_info.pack()
    def create_entry(self, text, default_val, row):
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=1)
        label = tk.Label(self.input_frame, text=text, font=("Helvetica", 12, "bold"), fg="white", bg="#8968CD", anchor="w")
        label.grid(row=row*2, column=0, padx=10, pady=5, sticky="ew")
        entry = tk.Entry(self.input_frame, font=("Helvetica", 12, "bold italic"), fg="#4B0082", bg="#FFF0F5", width=10, bd=2, relief=tk.GROOVE)
        entry.insert(0, f"{default_val:.3f}")
        entry.grid(row=row*2, column=1, padx=10, pady=5, sticky="se")
        setattr(self, f'entry_{row}', entry)
    def get_entry_values(self):
        try:
            d1 = float(self.entry_1.get())
            d2 = float(self.entry_3.get())
            d3 = float(self.entry_5.get())
            d4 = float(self.entry_7.get())
            d5 = float(self.entry_9.get())
            return d1, d2, d3, d4, d5
        except ValueError as ve:
            print("Error: Invalid data format")
            print("Error:", ve)
            return None
    
    def calculate_G_Bagging_regressor(self):
        values = self.get_entry_values()
        if values is None:
            self.br_output_text_gradientbaggingregressor.delete(1.0, tk.END)
            self.br_output_text_gradientbaggingregressor.insert(tk.END, "Error: Invalid input values")
            return
        d1, d2, d3, d4, d5 = values
        try:
            base_dir = r"C:\Users\MUHAMMAD SAUD KHAN\Documents\Waleed\software-and-prediction-main\V2 GUI for Predicting Compressive Strength of Roller Compacted Concrete (GBR)\GUI for Predicting Compressive Strength of Roller Compacted Concrete (GBR)"
            filename = r"RCC.xlsx"
            df = pd.read_excel(f"{base_dir}/{filename}")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            print(f"Data split into inputs and outputs:\nx shape: {x.shape}, y shape: {y.shape}")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=500)
            regressor= GradientBoostingRegressor(n_estimators = 60, max_depth=5, min_samples_split=8, min_samples_leaf=1,  learning_rate = 0.1)
            regressor.fit(x_train, y_train)
            input_data = np.array([d1, d2, d3, d4, d5]).reshape(1, -1)
            y_pred = regressor.predict(input_data)
            self.br_output_text_gradientbaggingregressor.delete(1.0, tk.END)
            self.br_output_text_gradientbaggingregressor.insert(tk.END, f"{y_pred[0]:.4f}")
            self.br_output_text_gradientbaggingregressor.config(font=("Helvetica", 12, "bold"), foreground="#E30B5D")

        except Exception as e:
            print(f"An error occurred: {e}")
            self.br_output_text_gradientbaggingregressor.delete(1.0, tk.END)
            self.br_output_text_gradientbaggingregressor.insert(tk.END, f"Error: {str(e)}")
    def predict(self):
        self.calculate_G_Bagging_regressor()
    def clear_fields(self):
        for i in range(1, 10, 2):
            entry = getattr(self, f'entry_{i}', None)
            if entry:
                entry.delete(0, tk.END)
        self.br_output_text_gradientbaggingregressor.delete(1.0, tk.END)
if __name__ == "__main__":
    root = tk.Tk()
    gui = RangeInputGUI(root)
    root.mainloop()