import numpy as np
import pandas as pd
import glob
import os
from utils import findDay, findSeason, findLastDayofMonth, correlation


dataset_folder = "../datasets/"
raw_dataset_address = dataset_folder + "data.csv"

# reading main dataset
Raw_dataset = pd.read_csv(raw_dataset_address, parse_dates=True)
# keeping only sales with correlation >= 0.8 with sales of next day
family_sales = Raw_dataset.drop_duplicates(['date', 'store_nbr'])
family_sales = family_sales[['date', 'store_nbr']]
for family in ['DAIRY', 'BEVERAGES', 'BREAD/BAKERY', 'POULTRY', 'PREPARED FOODS', 'SEAFOOD']:
    family_sales[family + '_sales'] = Raw_dataset[Raw_dataset['family'] == family]['sales'].tolist()
    family_sales[family + '_promotion'] = Raw_dataset[Raw_dataset['family'] == family]['onpromotion'].tolist()

for store in Raw_dataset['store_nbr'].unique():
    labels = family_sales[family_sales['store_nbr'] == store]['DAIRY_sales'][1:].tolist()
    labels.append(-1)
    family_sales.loc[family_sales['store_nbr'] == store, 'nextday_DAIRY_sales'] = labels

Main_dataset = family_sales[family_sales['nextday_DAIRY_sales'] >= 0]
# Main_dataset = Main_dataset[Main_dataset['family'] == "DAIRY"]
# Main_dataset = Main_dataset.drop(columns='family')
# splitting date feature into day of month, month, season, year, week day
Main_dataset['day_of_month'] = Main_dataset['date'].map(lambda x: int(x.split("-")[2]))
Main_dataset['month'] = Main_dataset['date'].map(lambda x: int(x.split("-")[1]))
Main_dataset['year'] = Main_dataset['date'].map(lambda x: int(x.split("-")[0]))
Main_dataset['season'] = Main_dataset['month'].apply(findSeason)
Main_dataset['week_day'] = Main_dataset['date'].apply(findDay)
# defining number of days before each wage day of month ==> wages in the public sector are paid every two weeks on
# the 15th and on the last day of the month
for year in Main_dataset['year'].unique():
    for month in Main_dataset['month'].unique():
        last_day = findLastDayofMonth(year, month)
        Main_dataset.loc[(Main_dataset['year'] == year) & (Main_dataset['month'] == month) & (
                Main_dataset['day_of_month'] <= 15), 'until_wage_day'] = 15 - Main_dataset.loc[
            (Main_dataset['year'] == year) & (Main_dataset['month'] == month) & (
                    Main_dataset['day_of_month'] <= 15), 'day_of_month']
        Main_dataset.loc[(Main_dataset['year'] == year) & (Main_dataset['month'] == month) & (
                Main_dataset['day_of_month'] > 15), 'until_wage_day'] = last_day - Main_dataset.loc[
            (Main_dataset['year'] == year) & (Main_dataset['month'] == month) & (
                    Main_dataset['day_of_month'] > 15), 'day_of_month']

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

    elif file_name == 'oil':
        Oil_dataset = pd.read_csv(file, parse_dates=True)
        Main_dataset = pd.merge(Main_dataset, Oil_dataset, how='left', on=['date'])
        del Oil_dataset

# Removing closed stores
Table_pivot = pd.pivot_table(Main_dataset[['store_nbr', 'year', 'DAIRY_sales']], aggfunc=np.sum, index='store_nbr',
                             columns='year')
cols = Table_pivot.columns
bt = Table_pivot.apply(lambda x: x > 0)
bt.apply(lambda x: list(cols[x.values]), axis=1)
closed_store_dict = {}
for year in cols:
    closed_stores = bt[bt[year] != True].index
    closed_store_dict[year[1]] = closed_stores.tolist()

for year in closed_store_dict:
    yearly_data = Main_dataset[Main_dataset['year'] == year]
    indexes = yearly_data[yearly_data['store_nbr'].isin(closed_store_dict[year])].index
    Main_dataset.drop(indexes, inplace=True)

# Removing year 2013
indexes = Main_dataset[Main_dataset['year'] == 2013].index
Main_dataset.drop(indexes, inplace=True)

# indexNames = Main_dataset[(Main_dataset['month'] == 1) & (Main_dataset['day_of_month'] == 1)].index
# Main_dataset.drop(indexNames, inplace=True)



# Drop low correlated features with nextday DAIRY sales
categorical_features = ['month', 'season', 'year', 'week_day', 'cluster', 'store_nbr', 'holiday']
low_corr_features = correlation(Main_dataset.drop(categorical_features, axis=1), 0.7)
Main_dataset = Main_dataset.drop(low_corr_features, axis=1)

# move label column to endof the columns
Main_dataset = Main_dataset[[c for c in Main_dataset if c not in ['nextday_DAIRY_sales']] + ['nextday_DAIRY_sales']]
Main_dataset.to_csv(dataset_folder + 'Integrated_data_RevisedClean_Families_HighCorrel.csv', index=False)
