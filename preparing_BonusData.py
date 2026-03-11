import numpy as np
import pandas as pd
import glob
import os
from utils import findDay, findSeason


dataset_folder = "../datasets/"
raw_dataset_address = dataset_folder + "bonusQ.csv"

# reading main dataset
Raw_dataset = pd.read_csv(raw_dataset_address, parse_dates=True)
# keeping only sales with correlation >= 0.8 with sales of next day
family_sales = Raw_dataset.drop_duplicates(['date', 'store_nbr'])
family_sales = family_sales[['date', 'store_nbr']]
for family in ['DAIRY', 'BEVERAGES', 'BREAD/BAKERY', 'POULTRY', 'PREPARED FOODS', 'SEAFOOD']:
    family_sales[family + '_sales'] = Raw_dataset[Raw_dataset['family'] == family]['sales'].tolist()

Main_dataset = family_sales

# splitting date feature into day of month, month, season, year, week day
Main_dataset['month'] = Main_dataset['date'].map(lambda x: int(x.split("-")[1]))
Main_dataset['year'] = Main_dataset['date'].map(lambda x: int(x.split("-")[0]))
Main_dataset['season'] = Main_dataset['month'].apply(findSeason)
Main_dataset['week_day'] = Main_dataset['date'].apply(findDay)

# Integrating more data from other dataset into main dataset
for file in glob.glob(dataset_folder + "*.csv"):
    file_name = os.path.basename(file).split(".")[0]
    if file == raw_dataset_address:
        continue
    elif file_name == "stores":
        Store_dataset = pd.read_csv(file)
        Main_dataset = pd.merge(Main_dataset, Store_dataset, how='left', on='store_nbr')
        del Store_dataset

    elif file_name == "holidays":
        Holiday_dataset = pd.read_csv(file, parse_dates=True)
        Holiday_dataset = Holiday_dataset[
            ((Holiday_dataset['type'] == "Holiday") | (Holiday_dataset['type'] == "Transfer"))
            & (Holiday_dataset['locale'] == "National")
            ]
        indexes = Holiday_dataset[Holiday_dataset['transferred']].index
        Holiday_dataset.drop(indexes, inplace=True)
        Main_dataset['holiday'] = np.False_
        for date in Holiday_dataset['date']:
            Main_dataset.loc[Main_dataset['date'] == date, 'holiday'] = True

        del Holiday_dataset

    elif file_name == 'transactions':
        Transaction_dataset = pd.read_csv(file, parse_dates=True)
        Main_dataset = pd.merge(Main_dataset, Transaction_dataset, how='left', on=['date', 'store_nbr'])
        del Transaction_dataset

Main_dataset.to_csv(dataset_folder + 'bonusQ_Integrated.csv', index=False)
