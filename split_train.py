import random

data_dir = "NLP_Dataset/MLHomework_Toxicity/val_.csv"


import pandas as pd
df = pd.read_csv(data_dir)

df_val = df.iloc[0:100]
df = df.iloc[100:]
df_val.to_csv("NLP_Dataset/MLHomework_Toxicity/val_1.csv")
df.to_csv("NLP_Dataset/MLHomework_Toxicity/train_1.csv")