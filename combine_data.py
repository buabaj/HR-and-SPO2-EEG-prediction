import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
os.chdir("C:/Users/eyadk/Desktop/dela/")

path = "C:/Users/eyadk/Desktop/dela/data2/"


def change_label(stress_label):
    if stress_label == "Relax":
        return 0
    else:
        return 1

data_list = []
for x in os.walk(path):
    if "Subject" in x[0]:
        print(x[0])
        subject_number = x[0].split("/")[6]
        directory = x[0]
        temp_file = x[2][0]
        hro2_file = x[2][1]

        temp_path = directory + "/" + temp_file
        hro2_path = directory + "/" + hro2_file

        df_temp = pd.read_csv(temp_path)
        df_hro2 = pd.read_csv(hro2_path)

        merged_data = pd.merge(df_hro2,df_temp,how='left',left_on=['Hour','Minute','Second'], right_on = ['Hour','Minute','Second'])
        merged_data["Subject_ID"] = subject_number

        data_list.append(merged_data)

##### Combining all datasets
full_data = pd.concat(data_list)
full_data["label"] = full_data.apply(lambda x: change_label(x["Label_x"]),axis=1)

##### Removing unnecessary columns
full_data = full_data.drop(["Label_x","Label_y","AccX","AccY","AccZ","EDA","Hour","Minute","Second"],axis=1)

##### Combining it with Subjects info
subjects_data = pd.read_excel("subjects_info.xlsx")
full_merged_data = pd.merge(full_data,subjects_data,how='left',on="Subject_ID")

full_merged_data.to_excel("full_merged_data.xlsx")

## class labels proprtions
pd.pivot_table(full_data, index="label",values="subject",aggfunc="count")


