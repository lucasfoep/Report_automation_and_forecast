# Imports libraries used
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from math import ceil
import calendar
import glob
import os
from pandas.tseries.offsets import DateOffset
import time

# Allows all rows from dataframes to be seen.
pd.set_option("display.max_rows", None)

start_time = time.time()

def column_date_name(df, a_str):

    # For each column
    for i in range(len(df.columns)):

        # If first five letters of column's name add up to 'Units'
        if df.columns[i][:5] == 'Units':

            # Formats the date string
            date_str = df.columns[i][5:].replace('/', '-')

            # Formats as datetime, then back to string, to get it in the right formatting.
            date_str = str(pd.to_datetime(date_str))[:-9]

            # Renames it
            df.rename(columns={df.columns[i]: a_str + ' ' + date_str},
                                 inplace=True)
            
    return df


def column_date_name_2(df, col_name, starting_date):
    '''Renames columns with dates'''
    
    # Iterates through all date columns
    for j in range(len(df.columns)):

        # If col_name equals Units
        if col_name == 'Units':
            
            # Store number of columns that are not Units related
            cols = 2
            
        # If col_name equals Sales
        else:
            
            # Store number of columns that are not Sales related
            cols = 3
        
        # If col_name is part of a column's name
        if col_name in df.columns[j]:

            # Formats a date offset as a string
            datestr = str(starting_date + DateOffset(months = j - cols))[:-9]
            
            # Rename
            df.rename(columns={df.columns[j]: 'Actual ' + col_name + ' ' + datestr},
                      inplace = True)
            
    # Returns edited dataframe
    return df


def calcprice(temp_df2, target_col, target_col_sum):
    '''Calculates the price. See main function description to understand how.'''
    
    # If target_col_sum is not zero
    if target_col_sum != 0:

        # Creates a column to store proportion of sales of a given part number
        temp_df2['%SalesLastMonth'] = temp_df2[target_col]/target_col_sum

        # Calculates a proportion of the price for each row, i.e part number for a given customer and stores it into a new column
        temp_df2['%Price'] = temp_df2['%SalesLastMonth'] * temp_df2['Price']

        # Sums up the proportional prices to find the weighed average price and stores it
        prop_sum = temp_df2['%Price'].sum()

        # Finds the average price for said part number of a given channel and stores it
        temp_df2['WAvgPrice'] = prop_sum
        
    # If target_col_sum is zero
    else:

        # Stores mean of existing price as the chosen price
        temp_df2['WAvgPrice'] = temp_df2['Price'].mean()
        
    # Returns temp_df2
    return temp_df2


def pricingfunc(df, channels, pns, target_col):
    '''Splits dataframe by channel and part number and calls function to calculate the price.'''
    
    # Creates an empty list to store dataframes
    df_list = []
    
    # For each channel in the list
    for channel in channels:

        # Creates a temporary dataframe with only said channel
        temp_df = df[df['Channel'] == channel]
        
        # Resets index of temporary dataframe
        temp_df.reset_index(drop = True, inplace = True)

        # For each part number in the list of part numbers
        for pn in pns:

            # Creates a temporary dataframe with only said part number
            temp_df2 = temp_df[temp_df['Part#'] == pn]

            # Resets index of temporary dataframe
            temp_df2.reset_index(drop = True, inplace = True)
            
            # Stores sum for target column in a variable
            target_col_sum = temp_df2[target_col].sum()

            # Calls function to calculate price
            temp_df2 = calcprice(temp_df2, target_col, target_col_sum)

            # Adds temp_df2 to a df list
            df_list.append(temp_df2)
                
    # Returns df_list
    return df_list


def mainpricingfunc(df, target_col):
    '''Uses last month sales to calculate a weighed mean price. If there are no sales for a given part number for last month
    it calculates a simple mean price for the part numbers for a given channel. The main function calls the sub functions.'''
    
    # Stores list of unique channels and part numbers for each dataframe
    channels = df['Channel'].unique().tolist()
    pns = df['Part#'].unique().tolist()
    
    # Calls pricing function to return an edited dataframe
    df_list = pricingfunc(df, channels, pns, target_col)
    
    # Returns df_list
    return df_list


def target_cols(target_date, actuals, a_string):
    
    # Creates an empty list
    actual_date_cols = []

    # Stores a first date
    first_actual = target_date - DateOffset(months = 7)

    # For each column name in the list of column names
    for actual in actuals:

        # Casts string to date
        actual_date = pd.to_datetime(actual[-10:])

        # If Units is part of the name of a given column, and its date is between first and target date
        if a_string in actual and actual_date < target_date and actual_date > first_actual:

            # Adds column name to list
            actual_date_cols.append(actual)
            
    return actual_date_cols


def to_float(df):
    '''Casts all columns, except for a few, to float'''
    
    # For each column of df
    for i in range(len(df.columns)):
    
        # If said column's name is not Channel, Customer Name or Part #
        if df.columns[i] not in ('Channel', 'Customer Name', 'Part#'):
            
            # Cast it to float
            df[df.columns[i]] = df[df.columns[i]].astype(float)
            
    # Returns updated dataframe
    return df


def split_dfs(df, channels):
    '''Splits dataframes by channel'''
    
    # Creates an empty list
    df_by_channel_list = []
    
    # For each channel in channels
    for channel in channels:

        # Filters dataframe by channel and stores in a temporary dataframe
        temp_df = df[df[df.columns[0]] == channel]
        
        # Resets temporary's dataframe index
        temp_df.reset_index(drop = True, inplace = True)
        
        # Adds temporary dataframe to list
        df_by_channel_list.append(temp_df)
        
    # Returns list
    return df_by_channel_list


def merge_dfs(list_1, list_2, cols_to_merge_on):
    '''Merges dataframes from two lists'''
    
    # Creates an empty list to be returned
    return_list = []
    
    # Iterates both lists from zero to length of list1. Both lists have same length
    for i in range(len(list_1)):

        # Merges two dataframes and stores it into a temporary dataframe
        temp_df = list_1[i].merge(list_2[i], how = 'left',
                                        on = cols_to_merge_on)

        # Adds temporary dataframe to return list
        return_list.append(temp_df)
    
    # Returns list
    return return_list


def remove_outliers(col):
    '''Calculates lower and upper limit fora given column to aid in removing outliers'''
    
    # Calculates third and first quartile for a given column
    q3, q1 = np.percentile(col, [75, 25])
    
    # Calculates the interquartile range for said column
    IQR = q3 - q1
    
    # Calculates the lower and upper limits for said column
    ll = q1 - 1.5 * IQR
    ul = q3 + 1.5 * IQR

    # Replaces values based on upper and lower limits
    col = np.where(col < ll, 0, np.where(col > ul, 1, col))
    
    # Returns column
    return pd.Series(col)


def calcproportion(temp_df2, target_col, target_col_sum, col_list):
    '''Calculates proportions and re-distributes Bookings and EDI. See main function description to understand how.'''
    
    # If target_col_sum is not zero
    if target_col_sum != 0:
        
        # Creates a column to store proportion of units sold of a given part number
        temp_df2['%UnitsLastMonth'] = temp_df2[target_col]/target_col_sum
        
        # Columns receive the product between themselves and the calculated proportions
        temp_df2[col_list] = temp_df2[col_list].multiply(temp_df2['%UnitsLastMonth'], axis="index")
                
    # If sum is zero, meaning no units sold for a given part number in a given channel
    else:
        
        # Columns receive the product between themselves and zero
        temp_df2[col_list] = temp_df2[col_list] * 0
        
    # Returns temp_df2
    return temp_df2


def proportionfunc(df, channels, pns, target_col, col_list):
    '''Splits dataframe by channel and part number and calls function to calculate proportions.'''
    
    # Creates an empty list to store dataframes
    df_list = []
    
    # For each channel in the list
    for channel in channels:

        # Creates a temporary dataframe with only said channel
        temp_df = df[df['Channel'] == channel]
        
        # Resets index of temporary dataframe
        temp_df.reset_index(drop = True, inplace = True)

        # For each part number in the list of part numbers
        for pn in pns:

            # Creates a temporary dataframe with only said part number
            temp_df2 = temp_df[temp_df['Part#'] == pn]

            # Resets index of temporary dataframe
            temp_df2.reset_index(drop = True, inplace = True)
            
            # Stores sum for target column in a variable
            target_col_sum = temp_df2[target_col].sum()

            # Calls function to calculate proportions
            temp_df2 = calcproportion(temp_df2, target_col, target_col_sum, col_list)

            # Adds temp_df2 to a df list
            df_list.append(temp_df2)
                
    # Returns df_list
    return df_list


def mainproportionfunc(df, target_col, col_list):
    '''Uses last month units sold to calculate proportions and re-distribute expected orders for a given part number amongst
    customers. The main function calls the sub functions.'''
    
    # Stores list of unique channels and part numbers for each dataframe
    channels = df['Channel'].unique().tolist()
    pns = df['Part#'].unique().tolist()
    
    # Calls pricing function to return an edited dataframe
    df_list = proportionfunc(df, channels, pns, target_col, col_list)
    
    # Returns df_list
    return df_list


def forecast_col_name(arange, astring, starting_date, df):
    '''Returns a dataframe with new columns, a weekly date_list and a monthly date_list'''
    
    # Creates empty lists
    date_list = []
    date_list_2 = []

    # Iterates the range passed
    for x in range(arange + 1): # Adds one to account for the first week
    
        # If on first iteration
        if x == 0:

            # Creates and stores a string to be used as a name for the first column
            strvar = astring + str(starting_date)[:-9]
            
            # Adds date to weekly date list
            date_list.append(starting_date)
            
            # Adds date to monthly date list
            date_list_2.append(starting_date)
            
            # Creates column and fills it with zeros
            df[strvar] = 0

        # If on any other iteration
        else:

            # Computes days or months to add based on the iteration
            days_to_add = (x * 7)
            months_to_add = (x * 1)

            # Creates a date and formats it as a string
            date = starting_date + timedelta(days = days_to_add)
            date_str = str(date)[:-9]
            date_2 = starting_date + relativedelta(months = + months_to_add)

            # Concatenates string passed and string date to build name of a given column
            strvar = astring + date_str
            
            # Adds date to their respective date lists
            date_list.append(date)
            date_list_2.append(date_2)

            # Creates column and fills it with zeros
            df[strvar] = 0
            
    # Returns edited dataframe and date lists
    return df, date_list, date_list_2


def same_month_days(date):
    '''Calculates how many dates in between two given weeks belong to the month at the beginning of first week'''
    
    # Formats it as string
    date_string = str(date)[:-9]
    
    # Splits string into year, month and day
    year, month, day = map(int, date_string.split("-"))

    # Finds number of days for a given month
    num_days = calendar.monthrange(year, month)[1]

    # If the difference between the number of days in a given month and the day of a given week is equal to or greater than
    # five
    if num_days - date.day + 1 >= 5:

        # Return five
        return 5

    # Otherwise
    else:

        # Returns the difference between the number of days of said month and the day of given week
        return num_days - date.day + 1
    
    
def num_of_weeks(adate):
    '''Returns the number of weeks in a given month'''
    
    # Stores year
    year = adate.year
    
    # Stores month
    month = adate.month
    
    # Calculates number of weeks
    weeksnum = len(calendar.monthcalendar(year, month))

    # Returns number of weeks
    return weeksnum
    
    
# Creates a function
def fill_rate_choice(date0, date1, fill_rate_list):
    '''Returns correct fill rate to be used'''
    
    # Keeps only months
    date0 = date0.month
    date1 = date1.month
    
    # Decides which fill rate to return based on the difference between the two months
    if date1 - date0 == 0:
        
        return fill_rate_list[0]
    
    elif date1 - date0 == 1:
        
        return fill_rate_list[1]
    
    elif date1 - date0 == 2:
        
        return fill_rate_list[2]
    
    elif date1 - date0 == 3:
        
        return fill_rate_list[3]
    
    elif date1 - date0 == 4:
        
        return fill_rate_list[4]
    
    else:
        
        return fill_rate_list[5]
    
    
def transpose(df, adate):
    
    # Transposes dataframe
    cht = df.T

    # For each column in the list of columns for the transpose dataframe
    for col in cht.columns:

        # Rename said column with its first value
        cht.rename(columns={col: cht[col][0]}, inplace=True)

    # Slices dataframe
    cht = cht[1:]

    # Creates an empty Date column
    cht['Date'] = ''

    # For each position of the index
    for i in range(len(cht.index)):

        # Date column receives date string from index in that position
        cht['Date'][i] = cht.index[i][-10:]

    # Resets index
    cht.reset_index(drop = True, inplace = True)

    # Reordering columns
    cht = cht[[cht.columns[-1]] + cht.columns[:-1].tolist()]

    # Filtering dataframe to keep only rows prior to current month
    cht = cht[pd.to_datetime(cht['Date']) < adate.replace(day = 1)]
    
    # returns edited dataframe
    return cht


def frconversion(frchoice):
    '''Converts fill rate string choices into a numerical value'''
    
    if frchoice == 'Current':
        
        return 0
    
    elif frchoice == '+1':
        
        return 1
    
    elif frchoice == '+2':
        
        return 2
    
    elif frchoice == '+3':
        
        return 3
    
    elif frchoice == '+4':
        
        return 4
    
    else:
        
        return 5


def forecastfunc(df, fill_rate_options, date_list, fill_rate_choice, fill_rate_period, frconversion):
    '''Calculates units and sales forecast and returns updated dataframe'''

    # For each date in the date list
    for adate in date_list:

        # Stores fill rate chosen by function. Second date_list varies
        frchoice = fill_rate_choice(date_list[0], adate, fill_rate_options)

        # Stores columns
        bookings = df['Units Booked ' + str(adate)[:-9]]
        edi = df['EDI Units ' + str(adate)[:-9]]
        fill_rate = df[frchoice]
        average = df['WeeklySixMonthsAvg ' + str(adate)[:-9]]
        ratio = df['WeeklyRatio ' + str(adate)[:-9]]
        units_forecast = 'Units Forecast ' + str(adate)[:-9]
        sales_forecast = 'Sales Forecast ' + str(adate)[:-9]

        # If the fill rate period entered matches the current fill rate choice the function incorporates averages on the forecast
        if frconversion(fill_rate_period) <= frconversion(frchoice):
        
            # Stores conditions
            conditions = [bookings != 0,
                         edi != 0,
                         average != 0]

            # Stores choices
            choices = [bookings * ratio,
                      edi * fill_rate * ratio,
                      average * ratio]
            
        # Otherwise, it doesn't
        else:
            
            # Stores conditions
            conditions = [bookings != 0,
                         edi != 0]

            # Stores choices
            choices = [bookings * ratio,
                      edi * fill_rate * ratio]

        # Forecast
        df[units_forecast] = np.select(conditions, choices) # Receives units forecast
        df[sales_forecast] = df[units_forecast] * df['Price'] # Receives sales forecast
        
    # Returns the updated dataframe
    return df


# Gets current work directory
cwd = os.getcwd()

# Gathers files' names.
filenames = glob.glob(cwd + "\*.xlsx")

# Imports spreadsheet into dataframe.
weekly = pd.read_excel(filenames[0], engine='openpyxl', sheet_name='ProjectionsPlus', usecols='B, J, Z:BX')

# Slicing dataframe to keep only rows with real data
weekly = weekly[3:]

# Keeping only rows that have Booked Open SO
weekly_booked = weekly[weekly[weekly.columns[0]] == 'BOOKING (CONSUMED)']

# Sorting by part number
weekly_booked.sort_values(weekly_booked.columns[1], inplace = True)

# Reseting index
weekly_booked.reset_index(drop = True, inplace = True)

# Dropping duplicates
weekly_booked.drop_duplicates(subset = 'Part#', keep = 'first', inplace = True)

# Calling function
weekly_booked = column_date_name(weekly_booked, 'Units Booked')

# Keeping only rows that have EDI FORECAST CURRENT
weekly_edi = weekly[weekly[weekly.columns[0]] == 'EDI FORECAST CURRENT']

# Sorting by part number
weekly_edi.sort_values(weekly_edi.columns[1], inplace = True)

# Reseting index
weekly_edi.reset_index(drop = True, inplace = True)

# Dropping duplicates
weekly_edi.drop_duplicates(subset = 'Part#', keep = 'first', inplace = True)

# Calling function
weekly_edi = column_date_name(weekly_edi, 'EDI Units')

# Dropping first column of both dataframes
weekly_edi.drop('F-From', axis=1, inplace = True)
weekly_booked.drop('F-From', axis=1, inplace = True)

# Merging both dataframes
weekly = weekly_edi.merge(weekly_booked, how = 'outer', on = 'Part#')

# Filling NAs with zeros
weekly = weekly.fillna(0)

# Imports spreadsheet into dataframe.
sunits = pd.read_excel(filenames[1], engine='openpyxl', sheet_name='3YearSales',
                    usecols='C, H, AD:AO, ED:EO, ID:IO',
                       skiprows = 2)
# Slicing dataframe
sunits = sunits[2:-10]

# Reseting index
sunits.reset_index(drop = True, inplace = True)

# Imports spreadsheet into dataframe.
sdollars = pd.read_excel(filenames[1], engine='openpyxl', sheet_name='3YearSales',
                    usecols='C, E, H, AQ:BB, EQ:FB, IQ:JB, MD, ME',
                       skiprows = 2)

# Slicing dataframe
sdollars = sdollars[2:-10]

# Reseting index
sdollars.reset_index(drop = True, inplace = True)

# Renames columns of sdollars dataframe
sdollars.rename(columns={sdollars.columns[0]: 'Channel',
                         sdollars.columns[2]: 'Part#',
                         sdollars.columns[-2]:'List Price',
                         sdollars.columns[-1]:'Contract Price'},
                inplace = True)

# Renames first and second columns of sunits dataframe
sunits.rename(columns={sunits.columns[0]: 'Channel',
                      sunits.columns[1]: 'Part#'},
              inplace = True)

# Reads value from cel
starting_year = filenames[1:][0][-22:-18]

# Stores date
starting_date = pd.to_datetime(starting_year, format = '%Y')

# Calls function to edit dataframes
sunits = column_date_name_2(sunits, 'Units', starting_date)
sdollars = column_date_name_2(sdollars, 'Sales', starting_date)

# Creates a column to store prices chosen between list and contract prices
sdollars['Price'] = 0

# Creates sets of conditions
inter_conditions = (sdollars['Channel'] == 'ZZZ Unclas')|(sdollars['Channel'] == 'Internatio')
am_conditions = (sdollars['Channel'] == 'AM BULK/MX')|(sdollars['Channel'] == 'AM.Bulk')|(sdollars['Channel'] == 'AM.Kits')

# Replaces channels 860 and ZZZ Unclas by OE and Intercompany/Internatio, respectively.
sdollars['Channel'] = np.where(sdollars['Channel'] == 'CLASS 860', 'OE', sdollars['Channel'])
sdollars['Channel'] = np.where(inter_conditions, 'Intercompany/Internatio', sdollars['Channel'])
sdollars['Channel'] = np.where(am_conditions, 'Aftermarket', sdollars['Channel'])
sdollars['Channel'] = np.where(sdollars['Channel'] == 'SLP Intern', 'OE MEXICO', sdollars['Channel'])

sunits['Channel'] = np.where(sunits['Channel'] == 'CLASS 860', 'OE', sunits['Channel'])
sunits['Channel'] = np.where(inter_conditions, 'Intercompany/Internatio', sunits['Channel'])
sunits['Channel'] = np.where(am_conditions, 'Aftermarket', sunits['Channel'])
sunits['Channel'] = np.where(sunits['Channel'] == 'SLP Intern', 'OE MEXICO', sunits['Channel'])

# Creates and stores a series of conditions for both list and contract prices
list_conditions = (sdollars['Channel'] == 'OE')|(sdollars['Channel'] == 'OE MEXICO')|(sdollars['Channel'] == 'OES')|(sdollars['Channel'] == 'OES MEXICO')
contract_conditions = (sdollars['Channel'] == 'Aftermarket')|(sdollars['Channel'] == 'Rings.300')|(sdollars['Channel'] == 'Intercompany/Internatio')

# If conditions are met, price column receives list price
sdollars['Price'] = np.where(list_conditions, sdollars['List Price'], sdollars['Price'])

# If conditions are met, price column receives contract price
sdollars['Price'] = np.where(contract_conditions, sdollars['Contract Price'], sdollars['Price'])

# Sets conditions for when price is still zero
zero_price_list_conditions = (sdollars['Price'] == 0) & (sdollars['List Price'] != 0)
zero_price_contract_conditions = (sdollars['Price'] == 0) & (sdollars['Contract Price'] != 0)

# If price is still zero, price column receives List price
sdollars['Price'] = np.where(zero_price_list_conditions, sdollars['List Price'], sdollars['Price'])

# If price is still zero, price column receives contract price
sdollars['Price'] = np.where(zero_price_contract_conditions, sdollars['Contract Price'], sdollars['Price'])

# Dropping old price columns
sdollars.drop(columns = ['List Price', 'Contract Price'], inplace = True)

# Getting list of unique channels
unique_channels = sdollars['Channel'].unique().tolist()

# Creating empty list
unique_channels_filtered = []

# For each channel in unique channels
for channel in unique_channels:
    
    # If channel is not null
    if pd.isnull(channel) is False:
        
        # Add channel to list
        unique_channels_filtered.append(channel)
        
# Stores current date in a variable
current_date = pd.to_datetime(filenames[0][-45:-37]).replace(day=1)

# Creates list of dataframe's columns
cols = sdollars.columns.tolist()

# Creates an empty list for actual values
actuals = []

# Iterates over columns of dataframe
for col in cols:
    
    # If column's name contains the word Actual
    if 'Actual' in col:
        
        # Add it to list
        actuals.append(col)
        
# Calling function to generate sales target columns
sales_target_cols = target_cols(current_date, actuals, 'Sales')

# Creating a target column
sdollars['SixMonthsSum'] = sdollars[sales_target_cols].sum(axis = 1)

sales_target_col = 'SixMonthsSum'

# Calls function to calculate prices
df_list = mainpricingfunc(sdollars, sales_target_col)

# Creates an empty dataframe.
sdollars = pd.DataFrame()

# Iterates all dataframes from the df_list
for df in df_list:
    
    # Concatenates dataframes.
    sdollars = pd.concat([df, sdollars], ignore_index=True)
    
# Replacing NaNs with zeros
sdollars.fillna(0, inplace = True)

# Creates an empty list to store columns to be dropped
to_drop = ['SixMonthsSum']

# For each column in the list of columns for the dataframe
for col in sdollars.columns:
    
    # If Price or percentage sign is part of the name of said column, but the name of said column is not Price
    if ('Price' in col or '%' in col) and col != 'WAvgPrice':
        
        # Adds column name to the list of columns to be dropped
        to_drop.append(col)
        
# Drops columns from list
sdollars.drop(to_drop, axis = 1, inplace = True)

# Renaming column as Price
sdollars.rename(columns={'WAvgPrice': 'Price'}, inplace=True)

# Calls function to cast columns to float. This had to be done to avoid groupby removing columns.
sdollars = to_float(sdollars)
sunits = to_float(sunits)

# Grabbing only pricing column
sdollars_pricing = sdollars[[sdollars.columns[0], sdollars.columns[2], sdollars.columns[-1]]]

# Dropping pricing column and Customer Name
sdollars.drop([sdollars.columns[1], sdollars.columns[-1]], axis = 1, inplace = True)

# Groups rows by channel and part number
sunits = sunits.groupby(['Channel', 'Part#'], as_index = False).sum()
sdollars = sdollars.groupby(['Channel', 'Part#'], as_index = False).sum()
sdollars_pricing = sdollars_pricing.groupby(['Channel', 'Part#'], as_index = False).mean()

# Creating a copy of sdollars_pricing Price column.
# This had to be done this way so that Price columns could be grouped by mean and remaining columns by sum
sdollars['Price'] = sdollars_pricing[sdollars_pricing.columns[-1]]

# Creates list of unique channels for dataframes
channels = sdollars[sdollars.columns[0]].unique()

# Calling function to create lists
sunits_by_channel_list = split_dfs(sunits, channels)
sdollars_by_channel_list = split_dfs(sdollars, channels)

# Creating list of columns to merge on
cols_to_merge_on = [sdollars_by_channel_list[0].columns[0], sdollars_by_channel_list[0].columns[1]]

# Calling function to generate list of merged dataframes
sreport_list = merge_dfs(sunits_by_channel_list, sdollars_by_channel_list, cols_to_merge_on)

# Imports spreadsheet into dataframe.
frates = pd.read_excel(filenames[2], engine='openpyxl', usecols='B:H')

# Iterates all columns except for the first one
for col in range(1, len(frates.columns)):

    # Applies function to column
    frates[frates.columns[col]] = remove_outliers(frates[frates.columns[col]])
    
# Creating a list of repeated dataframes to be able to re-utilize merge_dfs function
fill_rate_list = []
weekly_list = []

# For the length of the sreport_list
for x in range(len(sreport_list)):
    
    # Add frates to fill_rate_list and weekly to weekly_list
    fill_rate_list.append(frates)
    weekly_list.append(weekly)
    
# Creates list of columns to merge on
cols_to_merge_on = [weekly_list[0].columns[0]]

# Calls function to generate list of merged dataframes
sreport_weekly = merge_dfs(sreport_list, weekly_list, cols_to_merge_on)

# Creates list of columns to merge on
cols_to_merge_on = [fill_rate_list[0].columns[0]]

# Calls function to generate list of merged dataframes
sreport = merge_dfs(sreport_weekly, fill_rate_list, cols_to_merge_on)

# Creates an empty dataframe.
final_report = pd.DataFrame()

# For each dataframe in sreport
for y in range(len(sreport)):
    
    # Concatenates final report and dataframe in sreport
    final_report = pd.concat([sreport[y], final_report], ignore_index = True)
    
# Creates an empty list
edi_booked_cols = []

# For each column name in the list of columns for the final_report df
for col in final_report.columns:
    
    # If the name of the column contains either EDI or Booked
    if 'EDI' in col or 'Booked' in col:
        
        # Add column name to list
        edi_booked_cols.append(col)
        
# Creates list of dataframe's columns
cols = final_report.columns.tolist()

# Creates an empty list for actual values
actuals = []

# Iterates over columns of dataframe
for col in cols:
    
    # If column's name contains the word Actual
    if 'Actual' in col:
        
        # Add it to list
        actuals.append(col)
        
# Calling function to generate list of target columns for units sold
units_target_cols = target_cols(current_date, actuals, 'Units')

# Creating a column to store six months sum worth of units sold
final_report['SixMonthsSum'] = final_report[units_target_cols].sum(axis = 1)

# Creating a target column
units_target_col = 'SixMonthsSum'

# Calling function
df_list = mainproportionfunc(final_report, units_target_col, edi_booked_cols)

# Creates an empty dataframe.
final_report = pd.DataFrame()

# Iterates all dataframes from the df_list
for df in df_list:
    
    # Concatenates dataframes.
    final_report = pd.concat([df, final_report], ignore_index=True)
    
# Replacing NaNs with zeros
final_report.fillna(0, inplace = True)

# Dropping last two columns of dataframe.
final_report = final_report[final_report.columns[:-2]]

# Replacing NaNs with zeros
final_report.fillna(0, inplace=True)

# Sorting it
final_report.sort_values(by = [final_report.columns[0], final_report.columns[1]], ignore_index = True,
                        inplace = True)

# Assuming this is run on a Monday, otherwise starting date might need adjustments to make sure it is a Monday date.
# Gets starting date from file name
starting_date = pd.to_datetime(filenames[0][-45:][:8])

# Adds six months to it to find ending date
ending_date = starting_date + relativedelta(months = + 6)

# Calculates the number of days in between the two dates
days = ((ending_date - starting_date)).days

# Calculates the number of weeks
weeks = ceil(days/7)

# Looking at it
weeks

# Calls function
final_report_2, date_list, date_list_2 = forecast_col_name(weeks, 'Units Forecast ', starting_date, final_report)
final_report_2, date_list, date_list_2 = forecast_col_name(weeks, 'Sales Forecast ', starting_date, final_report)

# Setting date to be the first date of the list, i.e, date when report is being run.
adate = date_list[0]

# Creates a date string to access column
new_date = str(adate + timedelta(days = 14))[:-9]

# Concatenates date string with rest of column name
col_name = 'EDI Units ' + new_date

# Finds and stores index of given part number
idx = final_report_2.loc[(final_report_2['Part#'] == '807311R')].index[0]

# Uses index and column name to pass an NaN value to cell
final_report_2.at[idx, col_name] = np.nan

# Creates a copy of the dataframe with only numeric values
df_interpolate = final_report_2[final_report_2.columns[2:]]

# Making sure columns are of the float type
df_interpolate = to_float(df_interpolate)

# Applies linear interpolation to the only NaN in the dataframe
df_interpolate.interpolate(method ='linear', axis = 1, limit = 1, limit_direction ='forward', inplace = True)

# Copy the value interpolated into the original dataframe
final_report_2.at[idx, col_name] = df_interpolate.at[idx, col_name]

# Storing date for last actual volumes
last_actual = pd.to_datetime(actuals[-1][-10:])

# Stores names of first two columns in a list
first_two = final_report_2.columns[:2].tolist()

# Creates an empty list to store all other columns
all_other_cols = []

# For each column in the list of columns from final_report_2
for col in final_report_2.columns:
    
    # If the word Actual is not part of the name of said column
    if 'Actual' not in col:
        
        # Add column name to list of all other columns
        all_other_cols.append(col)
        
# Split final_report_2 in two dataframes with different columns, keeping Channel and Part# on both so that they can
# be merged back together
df1 = final_report_2[first_two + actuals]
df2 = final_report_2[all_other_cols]

# Grouping first dataframe using sum and second dataframe using mean
df1 = df1.groupby(['Channel', 'Part#'], as_index = False).sum()
df2 = df2.groupby(['Channel', 'Part#'], as_index = False).mean()

# Merging dataframes back together
final_report_2 = df1.merge(df2, how = 'outer', on = ['Channel', 'Part#'])

# Creates empty list
actual_units = []
actual_sales = []

# For each column name in the actuals list
for col_name in actuals:
    
    # If the word Units is part of the name
    if 'Units' in col_name:
        
        # Add that column name to the actual_units list
        actual_units.append(col_name)
        
    # Otherwise, this is an actual sale number
    else:
        
        # Add that column name to the actual_sales list
        actual_sales.append(col_name)
        
# Creates a copy of the dataframe
tseries = final_report_2.copy()

# Concatenates Channel and Part#
tseries['Part#'] = tseries['Channel'] + '-' + tseries['Part#']

# Removes the Channel column
tseries = tseries[tseries.columns[1:]]

# Splits dataframe into units and sales
utseries = tseries[['Part#'] + actual_units]
stseries = tseries[['Part#'] + actual_sales]

# Groups rows by channel
channel_history = final_report_2.groupby(['Channel'], as_index = False).sum()

# Stores column name into variable
channel = ['Channel']

# Creates list of columns
cs_units_cols = (channel + actual_units)
cs_sales_cols = (channel + actual_sales)

# Updates dataframe's clumns
channel_history_units = channel_history[cs_units_cols]
channel_history_sales = channel_history[cs_sales_cols]

# Calling function
chtu = transpose(channel_history_units, adate)
chts = transpose(channel_history_sales, adate)
utseries = transpose(utseries, adate)
stseries = transpose(stseries, adate)

# Exporting it to a csv file
chtu.to_csv('Channel_history_units.csv', index = False)
chts.to_csv('Channel_history_sales.csv', index = False)
utseries.to_csv('Historic_units_sold.csv', index = False)
stseries.to_csv('Historic_Sales.csv', index = False)

avg_col_names = [] # Creates list to store column names that will store six months averages

for i in range(6):
    
    avgdate = (adate + DateOffset(months = i)).replace(day=1)
    avgcolname = 'SixMonthsAvg ' + str(avgdate)[:-9]
    avg_col_names.append(avgcolname)

# For each column name in the list of column names
for col_name in avg_col_names:
    
    # Insert new column, named after column name, into dataframe
    final_report_2.insert(len(final_report_2.columns), col_name, np.zeros(len(final_report_2)).tolist(), True)
    
# adate.replace(day=1) replaces day of date with one
# -DateOffset(months = 6) finds date six months prior
firstdate = adate.replace(day=1) - DateOffset(months = 6) # Storing first date for average

# Finding and storing subsequent dates
seconddate = firstdate + DateOffset(months = 1)
thirddate = seconddate + DateOffset(months = 1)
fourthdate = thirddate + DateOffset(months = 1)
fifthdate = fourthdate + DateOffset(months = 1)
sixthdate = fifthdate + DateOffset(months = 1)

# Storing column names
first_col_name = 'Actual Units ' + str(firstdate)[:-9]
second_col_name = 'Actual Units ' + str(seconddate)[:-9]
third_col_name = 'Actual Units ' + str(thirddate)[:-9]
fourth_col_name = 'Actual Units ' + str(fourthdate)[:-9]
fifth_col_name = 'Actual Units ' + str(fifthdate)[:-9]
sixth_col_name = 'Actual Units ' + str(sixthdate)[:-9]

# Adding names to a list
six_col_names = [first_col_name, second_col_name, third_col_name, fourth_col_name, fifth_col_name, sixth_col_name]

# From zero to five
for i in range(6):
    
    # Remove first item of list
    six_col_names = six_col_names[1:]
    
    # Adds average column name to list
    six_col_names.append(final_report_2.columns[-6 + i])
    
    # Calculating averages
    final_report_2[final_report_2.columns[-6 + i]] = final_report_2[six_col_names].sum(axis=1)/6
    
# For each date in the list of dates
for d in date_list:
    
    weekly_avg_col_name = 'WeeklySixMonthsAvg ' + str(d)[:-9]
    
    # Insert new column, named after column name, into dataframe
    final_report_2.insert(len(final_report_2.columns), weekly_avg_col_name, np.zeros(len(final_report_2)), True)
    
# Creates an empty list to store name of weekly average columns
weeklyavgs = []

# For each column in the list of columns for the dataframe
for col in final_report_2.columns:
    
    # If WeeklySix is part of the columns name
    if 'WeeklySix' in col:
        
        # Adds it to list
        weeklyavgs.append(col)

# For each average column name
for col1 in avg_col_names:
    
    # Stores date
    avgdate = pd.to_datetime(col1[-10:])
    
    # Stores how many weeks said month has
    numberofweeks = num_of_weeks(avgdate)
    
    # For each one of the weekly average columns
    for col2 in weeklyavgs:
        
        # Stores date
        avgdate2 = pd.to_datetime(col2[-10:])
    
        # If month of monthly average equals month of weekly average
        if avgdate.month == avgdate2.month:
            
            # Weekly average column receives monthly average column divided by the number of weeks of said month
            final_report_2[col2] = final_report_2[col1]/numberofweeks

# Starts a variable
remainder = 0

# For each date in the date_list
for d in date_list:
    
    # Finds how many days of a given week belong to the month of the beginning of the week
    dayss = same_month_days(d)

    # Calculates a ratio of how many days are from said month versus how many days are not from the same month
    # and adds a remainder to it, in case there are days from previous week that belong to this month
    ratio = dayss/5 + remainder
    
    # Creates column name
    weekly_ratio_col_name = 'WeeklyRatio ' + str(d)[:-9]
    
    # Creates and inserts new column, named after column name, into dataframe and populates it with the calculated ratio
    final_report_2.insert(len(final_report_2.columns), weekly_ratio_col_name, np.full((1, len(final_report_2)), ratio)[0], True)
    
    # Re-starts remainder
    remainder = 0
    
    # If ratio is smaller than one, i.e if at least one day of said week does not belong to month of said week
    if ratio < 1:
        
        # Calculate a remainder ratio
        remainder = 1 - ratio
        
# Splitting dataframe by channels
df_list = split_dfs(final_report_2, channels)

# Creates a list of fill rate options
fill_rate_options = ['Current', '+1', '+2', '+3', '+4', '+5']

# Aftermarket
df_list[0] = forecastfunc(df_list[0], fill_rate_options, date_list, fill_rate_choice, '+1', frconversion)

# Intercompany/International
df_list[1] = forecastfunc(df_list[1], fill_rate_options, date_list, fill_rate_choice, 'Current', frconversion)

# OE
df_list[2] = forecastfunc(df_list[2], fill_rate_options, date_list, fill_rate_choice, '+2', frconversion)

# OE Mexico
df_list[3] = forecastfunc(df_list[3], fill_rate_options, date_list, fill_rate_choice, 'Current', frconversion)

# OES
df_list[4] = forecastfunc(df_list[4], fill_rate_options, date_list, fill_rate_choice, '+2', frconversion)

# OES Mexico
df_list[5] = forecastfunc(df_list[5], fill_rate_options, date_list, fill_rate_choice, 'Current', frconversion)

# Rings
df_list[6] = forecastfunc(df_list[6], fill_rate_options, date_list, fill_rate_choice, '+1', frconversion)

# Creates an empty dataframe.
final_report_2 = pd.DataFrame()

# For each dataframe in sreport
for df in df_list:
    
    # Concatenates final report and dataframe in sreport
    final_report_2 = pd.concat([df, final_report_2], ignore_index = True)
    
# Sorting it
final_report_2.sort_values(by = [final_report.columns[0], final_report.columns[1]], ignore_index = True,
                        inplace = True)

# Storing current date
current_date = date_list[0]

# Stores target date in a variable
target_date = current_date.replace(day=1)

# Stores name of target column in a variable
as_target_col = 'Actual Sales ' + str(target_date)[:-9]
au_target_col = 'Actual Units ' + str(target_date)[:-9]

# Creates an empty list
forecast_cols = []

# Getting list of forecast columns
# Iterates all columns
for col in final_report_2.columns:
    
    # If column's name has Channel, Part# or Forecast in it
    if 'Channel' in col or 'Part#' in col or 'Forecast' in col:
        
        # Adds column to list
        forecast_cols.append(col)

# Stores dataframe
forecasts = final_report_2[forecast_cols]

# Iterates over all columns of dataframe
for col in forecasts.columns:
    
    # If Part# and Channel are not in the column's name
    if 'Part#' not in col and 'Channel' not in col:
    
        # Renames column without day in it
        forecasts.rename(columns={col: col[:-3]}, inplace=True)
        
# Grouping columns
monthly_forecasts = forecasts.groupby(by = forecasts.columns, axis = 1).sum()

# Storing current date
current_date = date_list[0]

# Stores target date in a variable
target_date = current_date.replace(day=1)

# Stores name of target column in a variable
sf_target_col = 'Sales Forecast ' + str(target_date)[:-12]
uf_target_col = 'Units Forecast ' + str(target_date)[:-12]

# Updates first month of forecasts by adding to it anything that has already been sold
monthly_forecasts[sf_target_col] = monthly_forecasts[sf_target_col] + final_report_2[as_target_col]
monthly_forecasts[uf_target_col] = monthly_forecasts[uf_target_col] + final_report_2[au_target_col]

# Creates list of unique channels for dataframes
channels = final_report_2[final_report_2.columns[0]].unique()

# Calling function to create lists of dataframes split by channel
monthly_forecasts_list = split_dfs(monthly_forecasts, channels)
final_report_2_list = split_dfs(final_report_2, channels)

# Creating a list of columns to merge on
cols_to_merge_on = final_report_2.columns[:2].tolist()

# Calling function to generate list of merged dataframes
final_report_list = merge_dfs(monthly_forecasts_list, final_report_2_list, cols_to_merge_on)

# Creates an empty dataframe.
final_report = pd.DataFrame()

# For each dataframe in final_report_list
for y in range(len(final_report_list)):
    
    # Concatenates final report and dataframe in final_report_list
    final_report = pd.concat([final_report_list[y], final_report], ignore_index = True)
    
# Creates a list with first two columns
firstcols = ['Channel', 'Part#']

# Creates a list for price values
price = ['Price']

# Creates an empty list for EDI columns
edi = []

# Creates list of dataframe's columns
cols = final_report.columns.tolist()

# Iterates over columns of dataframe
for col in cols:
    
    # If column's name contains word EDI or Booked
    if 'EDI' in col or 'Booked' in col:
        
        # Add it to list
        edi.append(col)

# Creates empty lists for weekly and monthly forecasts
weekly_units_forecast_list = []
monthly_units_forecast_list = []
weekly_sales_forecast_list = []
monthly_sales_forecast_list = []

# Iterates over columns of dataframe
for col in cols:
    
    # If column's name contains the words Units Forecast and tenth to last character equals two, i.e date string is 10
    # characters long, i.e includes day, thus is a weekly units forecast
    if 'Units Forecast' in col and col[-10:][0] == '2':
        
        # Add to list
        weekly_units_forecast_list.append(col)
        
    # Otherwise, if column's name contains the words Sales Forecast and tenth to last character equals two, i.e date string is
    # 10 characters long, i.e includes day, thus is a weekly sales forecast
    elif 'Sales Forecast' in col and col[-10:][0] == '2':
        
        # Add to list
        weekly_sales_forecast_list.append(col)
        
    # Otherwise, if column's name contains the words Units Forecast and tenth to last character does not equal two, i.e this is
    # a monthly units forecast
    elif 'Units Forecast' in col and col[-10:][0] != '2':
        
        # Add to list
        monthly_units_forecast_list.append(col)
        
    # Otherwise, if column's name contains the words Sales Forecast and tenth to last character does not equal two, i.e this is
    # a monthly sales forecast
    elif 'Sales Forecast' in col and col[-10:][0] != '2':
        
        # Add to list
        monthly_sales_forecast_list.append(col)

# Sorts lists
weekly_units_forecast_list.sort()
weekly_sales_forecast_list.sort()


# Puts columns in the desired sequence
cols_to_df = (firstcols + actuals + price + edi + fill_rate_options + weekly_units_forecast_list + weekly_sales_forecast_list +
        monthly_units_forecast_list + monthly_sales_forecast_list)

# Passes new column sequence to dataframe
final_report = final_report[cols_to_df]

# Sorting df by first and second columns
final_report.sort_values(by = [final_report.columns[0], final_report.columns[1]], ignore_index = True,
                        inplace = True)

# Groups rows by channel
channel_summary = final_report.groupby(['Channel'], as_index = False).sum()

# Stores column name into variable
channel = ['Channel']

# Creates list of columns
cs_cols = (channel + monthly_sales_forecast_list)

# Updates dataframe's clumns
channel_summary = channel_summary[cs_cols]

# Stores date of report as a string
datestr = str(date_list[0])[:-9]

# Creates file name
file_name = 'Channel_summary_' + datestr + '.xlsx'

# Exporting to excel
channel_summary.to_excel(file_name, float_format = '%.2f')

# Creates file name
file_name = 'The_full_final_report_' + datestr + '.xlsx'

# Exporting to excel
final_report.to_excel(file_name, float_format = '%.2f')

end_time = time.time()
time_elapsed = end_time - start_time
print(f'Execution time was {time_elapsed}')
