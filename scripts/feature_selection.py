import pandas as pd
import numpy as np
import seaborn as sns
import os


df=pd.read_csv("data/processed/after_missing_value_imputation.csv")

export_df = df.drop(columns=["hdd",'weight', 'usb2',"everyday_use","performance","vga","multi_card_reader","antiglare","fingerprint_sensor","ethernet","hdmi","display_port","usb3"])

file_path="data/processed/after_feature_selection.csv"

if os.path.exists(file_path):
    os.remove(file_path)

export_df.to_csv(file_path,index=False)