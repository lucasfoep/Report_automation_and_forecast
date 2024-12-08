# Imports libraries used
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import calendar
import glob
import os
from pandas.tseries.offsets import DateOffset
import time
from openpyxl import load_workbook

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
    ''' The two main sheets, files 0 and 1, need to be merged. However, products
    are aggregated on file 0 and discriminated by customer on file 1. Products in
    file 1 must be aggregated to allow for the merge, but prices for one same product
    might vary per customer. This function uses last six months sales (summed into
    the target_col) to calculate a weighed average price accross all customers.
    If there are no sales for a given part number for last month it calculates a
    simple mean price for the part numbers. This is all done for each individual
    channel. The main function calls the sub functions.'''
    
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
        
        # Columns receive their average
        temp_df2[col_list] = temp_df2[col_list].mean()
        
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
    '''Uses last six months units sold to calculate proportions and re-distribute expected orders for a given part number amongst
    customers. The main function calls the sub functions.'''
    
    # Stores list of unique channels and part numbers for each dataframe
    channels = df['Channel'].unique().tolist()
    pns = df['Part#'].unique().tolist()
    
    # Calls function to return an edited dataframe
    df_list = proportionfunc(df, channels, pns, target_col, col_list)
    
    # Returns df_list
    return df_list


def forecast_col_name(df, date_list, astring):
    '''Returns a dataframe with new columns'''
    
    # For each date in the date list
    for adate in date_list:
        
        # Creates and stores a string to be used as a name for a given column
        strvar = astring + str(adate)[:-9]

        # Creates column and fills it with zeros
        df[strvar] = 0
        
    # Returns edited dataframe
    return df


def same_month_days(adate):
    '''Calculates how many dates in between two given weeks belong to the month at the beginning of first week'''
    
    # Formats it as string
    date_string = str(adate)[:-9]
    
    # Splits string into year, month and day
    year, month, day = map(int, date_string.split("-"))

    # Finds number of days for a given month
    num_days = calendar.monthrange(year, month)[1]

    # If the difference between the number of days in a given month and the day of a given week is equal to or greater than
    # five
    if num_days - adate.day + 1 >= 5:

        # Return five
        return 5

    # Otherwise
    else:

        # Returns the difference between the number of days of said month and the day of given week
        return num_days - adate.day + 1
    
    
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
    
    # Decides which fill rate to return based on the difference between the two months
    if relativedelta(date0.replace(day=1), date1.replace(day=1)).months == 0:
        
        return fill_rate_list[0]
    
    elif relativedelta(date0.replace(day=1), date1.replace(day=1)).months == 1:
        
        return fill_rate_list[1]
    
    elif relativedelta(date0.replace(day=1), date1.replace(day=1)).months == 2:
        
        return fill_rate_list[2]
    
    elif relativedelta(date0.replace(day=1), date1.replace(day=1)).months == 3:
        
        return fill_rate_list[3]
    
    elif relativedelta(date0.replace(day=1), date1.replace(day=1)).months == 4:
        
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
    
    elif frchoice == '+5':
        
        return 5
    
    else:
        
        return 10


def conditionsandchoices(frconversion, fill_rate_period, frchoice, bookings, edi, ihs_forecast, average, ratio, fill_rate,
                         part_nums):
    '''Creates conditions and choices for np.select and returns it'''
    
    # If the fill rate period entered matches the current fill rate choice the function incorporates averages on the forecast
    if frconversion(fill_rate_period) <= frconversion(frchoice):
    
        # Stores conditions
        conditions = [bookings != 0,
                     (edi != 0) & (fill_rate != 0),
                     (edi != 0) & (fill_rate == 0), # If There is EDI but Fill Rate is zero, this is a new part number without enough data for a fill rate, therefore use an average fill rate of .85
                     (part_nums != '807806R') & (part_nums != '807807R') & (part_nums != '807854R') & (part_nums != '807865R') & (part_nums != '807311R') & (part_nums != '808052R') & (part_nums != '808062R') & (part_nums != '808063R') & (part_nums != '808153R') & (part_nums != '808176') & (part_nums != '807377R') & (part_nums != '807943') & (part_nums != '806997R') & (part_nums != '808077') & (ihs_forecast != 0),
                     (part_nums != '807806R') & (part_nums != '807807R') & (part_nums != '807854R') & (part_nums != '807865R') & (part_nums != '807311R') & (part_nums != '808052R') & (part_nums != '808062R') & (part_nums != '808063R') & (part_nums != '808153R') & (part_nums != '808176') & (part_nums != '807377R') & (part_nums != '807943') & (part_nums != '806997R') & (part_nums != '808077') & (average != 0)]

        # Stores choices
        choices = [bookings * ratio,
                  edi * ratio * fill_rate,
                  edi * ratio * .85,
                  ihs_forecast * ratio,
                  average * ratio]
        
        # Stores choices for following week
        next_choices = [bookings * (1 - ratio),
                        edi * (1 - ratio) * fill_rate,
                        edi * (1 - ratio) * .85,
                        ihs_forecast * (1 - ratio),
                        average * (1 - ratio)]
        
    # Otherwise, it doesn't
    else:
        
        # Stores conditions
        conditions = [bookings != 0,
                     (edi != 0) & (fill_rate != 0),
                     (edi != 0) & (fill_rate == 0)]

        # Stores choices
        choices = [bookings * ratio,
                  edi * ratio * fill_rate,
                  edi * ratio * .85]
        
        # Stores choices for following week in case there is any remaining forecast
        # i.e forecast for current week includes days for following month
        next_choices = [bookings * (1 - ratio),
                        edi * (1 - ratio) * fill_rate,
                        edi * (1 - ratio) * .85]

    # Returns conditions and choices for np.select
    return conditions, choices, next_choices


def forecastfunc(df, fill_rate_options, date_list, fill_rate_choice, fill_rate_period, frconversion, conditionsandchoices):
    '''Calculates units and sales forecast and returns updated dataframe'''

    # Stores part number column in a variable, so that it is more legible

    part_nums = df['Part#']
    
    # For each date in the date list
    for adate in date_list:
        
        try:
        
            # Stores fill rate chosen by function. Second date_list varies
            frchoice = fill_rate_choice(adate, date_list[0], fill_rate_options)

            # Stores columns
            bookings = df['Units Booked ' + str(adate)[:-9]]
            edi = df['EDI Units ' + str(adate)[:-9]]
            fill_rate = df[frchoice]
            average = df['WeeklySixMonthsAvg ' + str(adate)[:-9]]
            ratio = df['WeeklyRatio ' + str(adate)[:-9]]
            units_forecast = 'Units Forecast ' + str(adate)[:-9]
            next_date = adate + DateOffset(weeks = 1)
            next_units_forecast = 'Units Forecast ' + str(next_date)[:-9]
            sales_forecast = 'Sales Forecast ' + str(adate)[:-9]
            ihs_forecast = df['IHS ' + str(adate)[:-9]]

            # Calls function to receive conditions and choices for np.select
            conditions, choices, next_choices = conditionsandchoices(frconversion,
                                                                     fill_rate_period,
                                                                     frchoice, bookings,
                                                                     edi, ihs_forecast,
                                                                     average,
                                                                     ratio,
                                                                     fill_rate,
                                                                     part_nums)

            # If next week is still within the forecasting range
            if next_date in date_list:

                # Assigns remaining of the week to next week due to days belonging to next month
                df[next_units_forecast] = np.select(conditions, next_choices) # Receives units forecast

            # Units forecast receives itself(in case it receieved a remainder from the previous week) plus forecast
            df[units_forecast] += np.select(conditions, choices) # Receives units forecast


            # After trimming the units forecasts, it calculates the sales forecasts
            df[sales_forecast] = df[units_forecast] * df['Price'] # Receives sales forecast
            
        except KeyError:
            
            print('KeyError')
        
    # Returns the updated dataframe
    return df


def vlookup(df1, df1_col1, df1_col2, df2, df2_col1, df2_col2):
    '''Looks up values from df1_col1 on df2_col1 and stores values from df2_col2 on df1_col2 when there is a match'''

    # Creates a list to store dataframes
    df1_list = []

    # Stores unique values from df1
    unique_df1_vals = df1[df1_col1].unique().tolist()

    # For each unique value
    for unique_val in unique_df1_vals:

        # Creates a temporary df1 with only one of the unique values
        temp_df1 = df1[df1[df1_col1] == unique_val]

        # Creates a temporary df2 for this given unique value
        temp_df2 = df2[df2[df2_col1] == unique_val]

        # Creates a record of how many rows there are for a given unique value on temp_df2
        records = len(temp_df2[df2_col2])

        # If df2_col2 is not empty
        if records != 0:

            # Stores value for all occurrences of given unique value in temp_df1
            temp_df1[df1_col2] = temp_df2[df2_col2].values[0]

        # Stores temp_df1 in a list
        df1_list.append(temp_df1)
        
    # Creates an empty dataframe in which to store partial dataframes from the df1_list
    df1 = pd.DataFrame()

    # For each dataframe in df1_list
    for partial_df1 in df1_list:

        # Concatenates a partial dataframe with df1
        df1 = pd.concat([partial_df1, df1], ignore_index = True)
    
    # Returns list of partial dataframes
    return df1


def selections(channel):
    '''Tkaes input for fill rate period variable, for the forecast function'''
    
    # Creates a counter
    counter = 0
    
    # Takes input from user
    fill_rate_period_entry = input(f'{channel}: ').lower().capitalize()
    
    # Forces loop until user enters a valid selection, updating counter to one.
    while counter == 0:

        # If entry is valid, update counter
        if fill_rate_period_entry in ('Current', '+1', '+2', '+3', '+4', '+5'):
            
            counter += 1
            
        # Otherwise, keep asking for valid input
        else:
            
            print('Your selection is not valid. Please make a valid selection.')
            fill_rate_period_entry = input(f'{channel}: ').lower().capitalize()
            
    # Return input
    return fill_rate_period_entry


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

# Get's year from file name
starting_year = filenames[1:][0][-22:-18]

# Stores date
starting_date = pd.to_datetime(starting_year, format = '%Y')

# Calls function to edit dataframes
sunits = column_date_name_2(sunits, 'Units', starting_date)
sdollars = column_date_name_2(sdollars, 'Sales', starting_date)

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

# Renames Contract Price column as Price
sdollars.rename(columns={'Contract Price': 'Price'}, inplace=True)

# Sets conditions for when price is still zero
zero_price_list_conditions = (sdollars['Price'] == 0) & (sdollars['List Price'] != 0)

# If price is still zero, price column receives List price
sdollars['Price'] = np.where(zero_price_list_conditions, sdollars['List Price'], sdollars['Price'])

# Dropping list price column
sdollars.drop(columns = ['List Price'], inplace = True)

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

# Creates a list to store columns to be dropped
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
frates = pd.read_excel(filenames[3], engine='openpyxl', usecols='B:H')
    
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

# Loads workbook
wb = load_workbook(filename = filenames[0], data_only = True, read_only = True)

# Stores sheet
ws = wb['ProjectionsPlus']

# Creates and stores temporary dates
temp_date = pd.to_datetime('01/01/21')
max_temp_date = pd.to_datetime('01/01/21')

# Creates an empty date list and a counter
date_list = []
counter = 0

# For each row in the sheet
for row in ws.rows:
    
    # For each cell in a given row
    for cell in row:
        
        # Try converting part of said cell to a date time object
        try:
            
            temp_date = pd.to_datetime(str(cell.value)[-8:])
            
        # If not possible, move on
        except:
            
            pass
        
        # If the temporary date is greater than the max temporary date
        if temp_date > max_temp_date:
            
            # Max temporary date receives temporary date
            max_temp_date = temp_date
            
            # Adds max temporary date to date list
            date_list.append(max_temp_date)
        
    # Updates counter
    counter += 1
    
    # If counter is greater or equal to one, i.e if on second row of sheet
    if counter >= 1:

        # Interrupt loop
        break

# Close the workbook after reading
wb.close()

# Calls function
final_report_2 = forecast_col_name(final_report, date_list, 'Units Forecast ')
final_report_2 = forecast_col_name(final_report, date_list, 'Sales Forecast ')

# Stores first date in a variable
adate = date_list[0]

# Creates a date string to access column
new_date = str(adate + timedelta(days = 14))[:-9]

# Concatenates date string with rest of column name
col_name = 'EDI Units ' + new_date

# Finds and stores index of given part number
idx = final_report_2.loc[(final_report_2['Part#'] == '807311R')].index[0]

# If value is not zero
if final_report_2.at[idx, col_name] != 0:

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


# For each date in the date_list
for d in date_list:
    
    # Finds how many days of a given week belong to the month of the beginning of the week
    dayss = same_month_days(d)

    # Calculates a ratio of how many days are from said month versus how many days are not from the same month
    ratio = dayss/5
    
    # Creates column name
    weekly_ratio_col_name = 'WeeklyRatio ' + str(d)[:-9]
    
    # Creates and inserts new column, named after column name, into dataframe and populates it with the calculated ratio
    final_report_2.insert(len(final_report_2.columns), weekly_ratio_col_name, np.full((1, len(final_report_2)), ratio)[0], True)
    
        

# Reading file
ihs = pd.read_excel(filenames[2], engine='openpyxl', usecols = 'F, M:Z', sheet_name='Detailed Forecast - Base Case', skiprows = 2)

# Renaming column so that it matches column from final_report_2
ihs.rename(columns={ihs.columns[1]: 'Part#'}, inplace=True)

# Storing year for both current and following year
current_year = str(adate.year)
next_year = str((adate + DateOffset(years = 1)).year)

# Creates list and stores first two columns in it
ihs_cols = [ihs.columns[0], ihs.columns[1]]

# For each column in the dataframe
for c in ihs.columns:
    
    # If either current or next year are part of a column's name
    if current_year in c or next_year in c:
        
        # Add column name to list
        ihs_cols.append(c)

# Keeps only relevant columns in the dataframe
ihs = ihs[ihs_cols]

# Creates empty list to store column names
ihs_col_names = []

# For each month of the year
for m in range(1, 13):

    # If it is a single unit month
    if len(str(m)) == 1:
        
        # Add a zero prior to said digit and stores it as a string
        ms = str(0) + str(m)
        
    # Otherwise
    else:
        
        # Just stores it as a string
        ms = str(m)
    
    # Creates name for a column and adds it to list
    ihs_col_names.append('IHS ' + current_year + '-' + ms)
    ihs_col_names.append('IHS ' + next_year + '-' + ms)
    
# Sorts list and stores it
ihs_col_names = sorted(ihs_col_names)

# Creates list with percentages to distribute revenue across months of the year
rev_rates = [0.08203, 0.0799, 0.094, 0.07801, 0.08239, 0.0864, 0.0818, 0.0927, 0.0838, 0.0909, 0.0772, 0.0709,
            0.08203, 0.0799, 0.094, 0.07801, 0.08239, 0.0864, 0.0818, 0.0927, 0.0838, 0.0909, 0.0772, 0.0709]

# From zero to 23, i.e each month of two years
for i in range(24):

    # Converts column name to date and stores it
    year = str(pd.to_datetime(ihs_col_names[i][-7:]).year)
    
    # If year matches current year
    if year == current_year:
        
        # Creates and inserts new column, named after column name, into dataframe and populates it with the calculated values
        # from the current year's column
        ihs.insert(len(ihs.columns), ihs_col_names[i], ihs[ihs.columns[2]] * np.full((1, len(ihs)), rev_rates[i])[0], True)
        
    # Otherwise
    else:
        
        # Creates and inserts new column, named after column name, into dataframe and populates it with the calculated values
        # from the next year's column
        ihs.insert(len(ihs.columns), ihs_col_names[i], ihs[ihs.columns[3]] * np.full((1, len(ihs)), rev_rates[i])[0], True)
    
# Drops columns with yearly values
ihs.drop([ihs.columns[2], ihs.columns[3]], axis = 1, inplace = True)

# Groups by first two columns
ihs = ihs.groupby(['Channel', 'Part#'], as_index = False).sum()

# Creates sets of conditions
ihs_am_conditions = (ihs['Channel'] == 'Bulk')|(ihs['Channel'] == 'Kits')
ihs_rings_condition = ihs['Channel'] == 'Rings'

# Replaces channels Bulk and Kits by Aftermarket.
ihs['Channel'] = np.where(ihs_am_conditions, 'Aftermarket', ihs['Channel'])
ihs['Channel'] = np.where(ihs_rings_condition, 'Rings.300', ihs['Channel'])

# Casting column to string
ihs['Part#'] = ihs['Part#'].astype(str)

# Finds intersection between IHS OE and OE MEXICO
ihsoe_to_oemex = list(set(ihs[ihs['Channel'] == 'OE']['Part#']) & set(final_report_2[final_report_2['Channel'] == 'OE MEXICO']['Part#']))

# Finds intersection between IHS OES and OES MEXICO
ihsoes_to_oesmex = list(set(ihs[ihs['Channel'] == 'OES']['Part#']) & set(final_report_2[final_report_2['Channel'] == 'OES MEXICO']['Part#']))

# Finds intersection between IHS OES and OE MEXICO
ihsoes_to_oemex = list(set(ihs[ihs['Channel'] == 'OES']['Part#']) & set(final_report_2[final_report_2['Channel'] == 'OE MEXICO']['Part#']))

# Finds intersection between IHS OE and OES
ihsoe_to_oes = list(set(ihs[ihs['Channel'] == 'OE']['Part#']) & set(final_report_2[final_report_2['Channel'] == 'OES']['Part#']))

# Updates channels
for p1 in ihsoe_to_oemex:

    ihs['Channel'] = np.where(ihs['Part#'] == p1, 'OE MEXICO', ihs['Channel'])
    
for p2 in ihsoes_to_oesmex:

    ihs['Channel'] = np.where(ihs['Part#'] == p2, 'OES MEXICO', ihs['Channel'])
    
for p3 in ihsoes_to_oemex:

    ihs['Channel'] = np.where(ihs['Part#'] == p3, 'OE MEXICO', ihs['Channel'])
    
for p4 in ihsoe_to_oes:

    ihs['Channel'] = np.where(ihs['Part#'] == p4, 'OES', ihs['Channel'])
    
# Removes rows that only have zeros on forecast columns
ihs = ihs.loc[~(ihs[ihs.columns[2:]] == 0).all(axis=1)]

# Re-sets index
ihs.reset_index(drop = True, inplace = True)

# Merges both data frames
final_report_2 = final_report_2.merge(ihs, how = 'outer', on = ['Channel', 'Part#'])

# Replaces NaNs with zeros
final_report_2.fillna(0, inplace = True)

# Creates empty list to store created weekly column names
weekly_ihs_col_names = []

# For each date in the date list
for d in date_list:
    
    # Creates a column name using date and adds it to list
    weekly_ihs_col_names.append('IHS ' + str(d)[:-9])
    
# For each column in the list of weekly column names
for c in weekly_ihs_col_names:
    
    # Insert it into data frame and fill with zeros
    final_report_2.insert(len(final_report_2.columns), c, 0, True)
    
# Creates empty lists for both weekly and monthly columns
ihs_monthly_cols = []
ihs_weekly_cols = []

# For each column in the list of columns
for col in final_report_2.columns:
    
    # If IHS is part of the name of said column
    if 'IHS' in col:
        
        # And if the length of the column name equals 11
        if len(col) == 11:
            
            # Adds it to the monthly columns list
            ihs_monthly_cols.append(col)
            
        # Otherwise
        else:
            
            # Adds it to the weekly list
            ihs_weekly_cols.append(col)
            
# For each column name in the monthly columns list
for cn1 in ihs_monthly_cols:
    
    # And for each column name in the weekly columns list
    for cn2 in ihs_weekly_cols:
        
        # Store date from column name one and two as date
        d1 = pd.to_datetime(cn1[-7:])
        d2 = pd.to_datetime(cn2[-10:])
        
        # If on both year and month
        condition1 = relativedelta(d1.replace(day=1), d2.replace(day=1)).years == 0
        condition2 = relativedelta(d1.replace(day=1), d2.replace(day=1)).months == 0
        
        if condition1 and condition2:

            # Weekly column receives value from monthy column divided by the number of weeks in a given month
            final_report_2[cn2] = final_report_2[cn1]/num_of_weeks(d1)
            
# Splitting dataframe by channels
df_list = split_dfs(final_report_2, channels)

# Creates a list of fill rate options
fill_rate_options = ['Current', '+1', '+2', '+3', '+4', '+5']

# Stores standard settings
fill_rate_period_aftermarket = '+1'
fill_rate_period_international = 'Current'
fill_rate_period_oe = '+2'
fill_rate_period_oe_mex = 'Current'
fill_rate_period_oes = '+2'
fill_rate_period_oes_mex = 'Current'
fill_rate_period_rings = '+1'

# Prints settings
print('')
print('='*100)
print('')
print('The current settings for when to use IHS or six-month averages are as follow:')
print('')
print('Aftermarket: +1')
print('Intercompany/International: Current')
print('OE: +2')
print('OE Mexico: Current')
print('OES: +2')
print('OES Mexico: Current')
print('Rings: +1')
print('')
print('='*100)

# Starts a counter, which will be used to force user to input valid entries
counter = 0

# Gets user's input
answer = str(input("Type 'yes' if you would like to change it. Otherwise, type 'no': ")).lower()

# Repeats loop until user picks a valid entry
while counter == 0:

    # If input is valid, counter is updated and breaks the loop
    if answer == 'yes' or answer == 'no':
        
        counter += 1
        
    # Otherwise, it keeps asking for a valid input
    else:
        
        print("You typed neither 'yes' nor 'no'.")
        answer = str(input("Type 'yes' if you would like to change it. Otherwise, type 'no': ")).lower()

# If answer is yes
if answer == 'yes':
    
    # Displays settings options
    print('')
    print("Your options are: 'Current', '+1', '+2', '+3', '+4', '+5'")
    print('')
    print('Make your selection:')
    
    # Gets settings from user
    fill_rate_period_aftermarket = selections('Aftermarket')
    fill_rate_period_international = selections('Intercompany/International')
    fill_rate_period_oe = selections('OE')
    fill_rate_period_oe_mex = selections('OE Mexico')
    fill_rate_period_oes = selections('OES')
    fill_rate_period_oes_mex = selections('OES Mexico')
    fill_rate_period_rings = selections('Rings')
    
    
# Aftermarket
df_list[0] = forecastfunc(df_list[0], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_aftermarket, frconversion, conditionsandchoices)

# Intercompany/International
df_list[1] = forecastfunc(df_list[1], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_international, frconversion, conditionsandchoices)

# OE
df_list[2] = forecastfunc(df_list[2], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_oe, frconversion, conditionsandchoices)

# OE Mexico
df_list[3] = forecastfunc(df_list[3], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_oe_mex, frconversion, conditionsandchoices)

# OES
df_list[4] = forecastfunc(df_list[4], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_oes, frconversion, conditionsandchoices)

# OES Mexico
df_list[5] = forecastfunc(df_list[5], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_oes_mex, frconversion, conditionsandchoices)

# Rings
df_list[6] = forecastfunc(df_list[6], fill_rate_options, date_list, fill_rate_choice, fill_rate_period_rings, frconversion, conditionsandchoices)

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

try:

    # Updates first month of forecasts by adding to it anything that has already been sold
    monthly_forecasts[sf_target_col] = monthly_forecasts[sf_target_col] + final_report_2[as_target_col]
    monthly_forecasts[uf_target_col] = monthly_forecasts[uf_target_col] + final_report_2[au_target_col]
    
except KeyError:
    
    print('KeyError')

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
channel_summary.to_excel(file_name, float_format = '%.2f', index = False)

# Creates file name
file_name = 'The_full_final_report_' + datestr + '.xlsx'

# Exporting to excel
final_report.to_excel(file_name, float_format = '%.2f', index = False)

# Creating and storing conditions
cond1 = final_report['Part#'] == '807311R'
cond2 = final_report['Part#'] == '808052R'
cond3 = final_report['Part#'] == '808103R'
cond4 = final_report['Part#'] == '807373'
cond5 = final_report['Part#'] == '808105'

# Using conditions to filter dataframe
frv = final_report[cond1 | cond2 | cond3 | cond4 | cond5]

# Re-setting index
frv.reset_index(drop = True, inplace = True)

# Creating column to group and concatenate by
frv['Description'] = 'FRV'

# Storing column names
units_fcst_col = monthly_units_forecast_list[0]
sales_fcst_col = monthly_sales_forecast_list[0]
sales_col = 'Actual Sales ' + str(adate.replace(day=1))[:-9]
units_col = 'Actual Units ' + str(adate.replace(day=1))[:-9]

# Keeping only subset of columns
frv = frv[['Description', units_fcst_col, sales_fcst_col, units_col, sales_col]]

# Grouping by
frv = frv.groupby('Description', as_index = False).sum()

# Creating new columns
frv['RemaningUnits'] = frv[frv.columns[1]] - frv[frv.columns[3]]
frv['RemaningDollars'] = frv[frv.columns[2]] - frv[frv.columns[4]]
frv['Remaning%U'] = frv[frv.columns[5]]/frv[frv.columns[1]]
frv['Remaning%D'] = frv[frv.columns[5]]/frv[frv.columns[1]]

# Removing first column
frv = frv[frv.columns[1:]]

# Splitting dataframe
frv_shipped = frv[[frv.columns[2], frv.columns[3]]]
frv_fcst = frv[frv.columns[0:2]]
frv_remaining = frv[[frv.columns[4], frv.columns[5]]]
frv_remaining_p = frv[[frv.columns[-1], frv.columns[-2]]]

# Renaming columns
frv_shipped.rename(columns = {frv_shipped.columns[0]: 'Units', frv_shipped.columns[1]: 'Dollars'}, inplace = True)
frv_fcst.rename(columns = {frv_fcst.columns[0]: 'Units', frv_fcst.columns[1]: 'Dollars'}, inplace = True)
frv_remaining.rename(columns = {frv_remaining.columns[0]: 'Units', frv_remaining.columns[1]: 'Dollars'}, inplace = True)
frv_remaining_p.rename(columns = {frv_remaining_p.columns[0]: 'Units', frv_remaining_p.columns[1]: 'Dollars'}, inplace = True)

# Concatenating dataframes
frv = pd.concat([frv_shipped, frv_fcst], ignore_index = True)
frv = pd.concat([frv, frv_remaining], ignore_index = True)
frv = pd.concat([frv, frv_remaining_p], ignore_index = True)

# Labeling indexes
frv.index = ['Shipped', 'FCST', 'Remaining', 'Remaining%']

# Creates file name
file_name = 'FRV_' + datestr + '.xlsx'

# Exporting to excel
frv.to_excel(file_name, float_format = '%.2f')

# Reading sheet into dataframe
descriptions = pd.read_excel(filenames[1], engine='openpyxl', sheet_name='3YearSales',
                    usecols='H:I', skiprows = 2)

# Removing first couple of rows
descriptions = descriptions[2:]

# Re-setting index
descriptions.reset_index(drop = True, inplace = True)

# Dropping duplicates
descriptions.drop_duplicates(inplace = True)

# Reading last forecast file
last_forecast = pd.read_excel(filenames[4], engine='openpyxl')

# Creates an empty dataframe
top_5 = pd.DataFrame()

# Storing name of column
current_sales_forecast = 'Sales Forecast ' + str(date_list[0].replace(day = 1))[:-12]

# Passing Part# column to new dataframe
top_5['Part#'] = final_report['Part#']

# Creating a description column on new dataframe to receive descriptions
top_5['Description'] = ''

# Calling function to update new dataframe
top_5 = vlookup(top_5, 'Part#', 'Description', descriptions, 'Part #', 'Part Description')

# Merging
top_5 = top_5.merge(final_report[['Part#', current_sales_forecast]], how = 'outer', on = 'Part#')

# Re-naming
top_5.rename(columns = {current_sales_forecast: 'Current ' + current_sales_forecast}, inplace=True)

# Merging
top_5 = top_5.merge(last_forecast[['Part#', current_sales_forecast]], how = 'outer', on = 'Part#')

# Re-naming
top_5.rename(columns = {current_sales_forecast: 'Last ' + current_sales_forecast}, inplace=True)

# Grouping by description and Part#
top_5 = top_5.groupby(['Part#', 'Description'], as_index = False).sum()

# Calculating the difference between the two
top_5['Last VS Current'] = top_5[top_5.columns[2]] - top_5[top_5.columns[3]]

# Calculating the relative difference
top_5['Last VS Current %'] = top_5['Last VS Current'] / top_5[top_5.columns[2]]

# Filling NaNs and replacing infinite values
top_5.fillna(0, inplace = True)
top_5.replace([np.inf, -np.inf], 0, inplace=True)

# Finding absolute value
top_5['Absolute'] = top_5['Last VS Current'].abs()

# Sorting rows by absolute values
top_5.sort_values(by = 'Absolute', ignore_index = True, inplace = True, ascending = False)

# Re-arranging columns and dropping absolute values
top_5 = top_5[['Description', 'Part#', top_5.columns[2], top_5.columns[3], 'Last VS Current', 'Last VS Current %']]

# Storing first 10 rows
top_5 = top_5.head(10)

# Creates file name
file_name = 'Top_5_' + datestr + '.xlsx'

# Exporting to excel
top_5.to_excel(file_name, float_format = '%.2f', index = False)


end_time = time.time()
time_elapsed = end_time - start_time
print(f'Execution time was {time_elapsed}')
