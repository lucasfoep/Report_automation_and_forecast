# Imports libraries used
import pandas as pd
import numpy as np
import glob
import os
from pandas.tseries.offsets import DateOffset
import time

# Stores a start time
start_time = time.time()

# Allows all rows from dataframes to be seen.
pd.set_option("display.max_rows", None)

# Gets current work directory
cwd = os.getcwd()

# Gathers files' names.
filenames = glob.glob(cwd + "\*.xlsx")

# Creates an empty dataframe.
df2 = pd.DataFrame()

# Iterates all files, except for the last one
for i in range(len(filenames) - 1):
    
    # Imports spreadsheet into dataframe.
    df1 = pd.read_excel(filenames[i], engine='openpyxl', sheet_name='ProjectionsPlus', usecols='B, D, J, K, Y:AJ')
    
    # df1.columns[0] returns name of first column
    # Filtering the dataframe by rows that have EDI FORECAST CURRENT on first column
    df1 = df1[df1[df1.columns[0]] == 'EDI FORECAST CURRENT']

    # df1.columns[2] returns name of third column
    # Sorting by part number
    df1.sort_values(df1.columns[2], inplace=True)

    # Reseting index
    df1.reset_index(drop = True, inplace = True)

    # Dropping duplicate part numbers
    df1.drop_duplicates(subset=df1.columns[2], keep='first', inplace=True)
    
    # Gets date from file name
    date_string = filenames[i][-38:][:8]
    
    # Stores date
    current_date = pd.to_datetime(date_string, format = '%Y%m%d').replace(day=1)
    
    # Adding date to label
    df1[df1.columns[0]] = df1[df1.columns[0]] + ' ' + str(current_date)[:-9]
    
    # Iterates through all date columns
    for j in range(4, len(df1.columns)):
        
        # If on first column
        if j == 4:
            
            # Renames first columns with full date
            df1.rename(columns={df1.columns[j]:current_date}, inplace = True)
            
        else:
            
            # Renames following columns by adding a month to each
            df1.rename(columns={df1.columns[j]:current_date + DateOffset(months = j - 4)}, inplace = True)    
    
    # Concatenates df1 and df2.
    df2 = pd.concat([df1, df2], ignore_index=True)
    
# Sorting by part number
df2.sort_values(df2.columns[2], inplace=True)

# Reseting index
df2.reset_index(drop = True, inplace = True)

# Replacing NAs with zeros
df2.fillna(0, inplace=True)

# Splitting dataframe in two so that columns can be sorted
labels = df2[df2.columns[:4]]
units = df2[df2.columns[4:]]

# Sorting DataFrame columns
units = units.reindex(columns = sorted(units.columns))

# Creating a temporary index so that dataframes can be put back together
labels['temp_index'] = np.arange(len(df2))
units['temp_index'] = np.arange(len(df2))

# Putting dataframes back together
df3 = labels.merge(units, how = 'left', on = 'temp_index')

# Dropping temporary index
df3.drop('temp_index', axis=1, inplace = True)

# Imports spreadsheet into dataframe.
df4 = pd.read_excel(filenames[-1], engine='openpyxl', sheet_name='3YearSales', usecols='H, I, AD:AO, ED:EO, ID:IO',
                    skiprows = 2)

# Renaming Part # column so that it matches df3's
df4.rename(columns={'Part #':'Part#'}, inplace = True)

# Slicing dataframe
df4 = df4[2:]

# Reads column as dataframe to get initial year. DO THIS WITH OPENPYXL, READ ONLY THE ONE CELL.
starting_year = pd.read_excel(filenames[-1], engine='openpyxl', sheet_name='3YearSales', usecols='AD', skiprows = 1)

# Keeping only the starting year
starting_year = starting_year.columns[0]

# Stores date
starting_date = pd.to_datetime(starting_year, format = '%Y')

# Iterates through all date columns
for k in range(2, len(df4.columns)):
        
    if k == 2:
            
        # Renames first columns with full date
        df4.rename(columns={df4.columns[k]:starting_date}, inplace = True)
            
    else:
            
        # Renames following columns by adding a month to each
        df4.rename(columns={df4.columns[k]:starting_date + DateOffset(months = k - 2)}, inplace = True)

# Getting number of colums to be dropped
to_drop = sum(df4.columns[2:] < df3.columns[4])

# Getting columns to be dropped
cols_to_drop = df4.columns[2:2 + to_drop]

# Dropping columns
df4.drop(cols_to_drop, axis=1, inplace = True)

# Creating columns
df4['F-From'] = 'Actual'
df4['Location'] = '-'

# Concatenating both dfs
df5 = pd.concat([df3, df4], ignore_index=True)

# Replacing NAs
df5.fillna(0, inplace=True)

# Sorting by part number
df5.sort_values(df5.columns[2], inplace=True)

# Reseting index
df5.reset_index(drop = True, inplace = True)

# Filtering
df1 = df1[df1[df1.columns[0]] == 'EDI FORECAST CURRENT']

# df5.columns[0] returns name of first column of df5
# df5[df5.columns[0]] returns first column of df5
# df3.columns[0] returns name of first column of df3
# df3[df3.columns[0]] returns first column of df3
# df5[df5.columns[0]].isin(df3[df3.columns[0]]) return a dataframe where what's on first column of df5 is on first column of df3
# [df5.columns[2]] returns third column of this new dataframe
# [df5.columns[2]].unique() returns only unique values for that column
# This creates a list of part numbers that have EDI FORECAST CURRENT
part_numbers = df5[df5[df5.columns[0]].isin(df3[df3.columns[0]])][df5.columns[2]].unique()

# Filtering df5 by items that have EDI FORECAST CURRENT
df5 = df5[df5[df5.columns[2]].isin(part_numbers)]

# Sorting by part number
df5.sort_values([df5.columns[2], df5.columns[0]], inplace=True)

# Reseting index
df5.reset_index(drop = True, inplace = True)

# Creates an empty variable
first_col_to_drop = ''

# Iterates through all date columns from df4
for c in range(2, len(df4.columns)):
    
    # If the sum of all values for given column is zero
    if df4[df4.columns[c]].sum() == 0:
        
        # Add column to list of columns to be dropped
        first_col_to_drop = df4.columns[c]
        
        break

# Creates a list of columns to be dropped
cols_to_drop_2 = []
        
# Iterates through columns from df5
for d in range(4, len(df5.columns)):
    
    # If there is a col to be dropped and column from df5 is greater than or
    # equal to first column to drop
    if len(str(first_col_to_drop)) > 0 and df5.columns[d] >= first_col_to_drop:
        
        # Add column from df5 to list of columns to be dropped
        cols_to_drop_2.append(df5.columns[d])
        
# Dropping columns
df5.drop(cols_to_drop_2, axis=1, inplace = True)

# Iterates over date columns from df5
for e in range(4, len(df5.columns)):
    
    # Creates a column with the date from one of the date columns plus an 'FR'
    df5[str(df5.columns[e])[:-8] + 'FR'] = 0

# Creates an empty dataframe.
df6 = pd.DataFrame()

# Stores number of fr columns
fr_cols = int((len(df5.columns) - 4)/2)

# Iterates over all unique part numbers
for z in range(len(part_numbers)):

    # Creates a temporary dataframe with only one part number
    temp_df = df5[df5[df5.columns[2]] == part_numbers[z]]
    
    # Reseting index
    temp_df.reset_index(drop = True, inplace = True)
    
    # Gets actuals
    temp_df_actuals = temp_df[temp_df[temp_df.columns[0]] == 'Actual']
    
    # Gets forecasts
    temp_df_forecasts = temp_df[temp_df[temp_df.columns[0]] != 'Actual']
    
    # Resets index of forecasts
    temp_df_forecasts.reset_index(drop = True, inplace = True)

    # Wanted to have done this in a vectorized manner, but zero values caused it to return an error
    # Iterates date columns
    for x in range(4, len(temp_df.columns) - fr_cols):

        actual = 0
        
        # Iterates rows of such columns
        for a in range(len(temp_df_actuals)):

            # Stores value from first row
            actual += temp_df_actuals[temp_df_actuals.columns[x]][a]
            idx = a

        for f in range(len(temp_df_forecasts)):
            
            # Stores value from each row
            projected = temp_df_forecasts[temp_df_forecasts.columns[x]][f]

            # If both actual and projected are different than zero
            if actual != 0 and projected != 0:

                # Inputs actual/projected to same row but at an FR column
                temp_df[temp_df.columns[x + fr_cols]][idx + f + 1] = actual/projected
                

    # Concatenates df6 and df5.
    df6 = pd.concat([df6, temp_df], ignore_index=True)
    
# Exporting to Excel.
df6.to_excel('Fill_rates.xlsx', float_format="%.2f")

# Creates an empty dataframe
df7 = pd.DataFrame()

# Iterates over all part numbers
for i in range(len(part_numbers)):

    # Creates a temporary dataframe with only a given part number
    temp_df = df6[df6[df6.columns[2]] == part_numbers[i]]
    
    # Reseting index
    temp_df.reset_index(drop = True, inplace = True)
    
    # Creates a temporary dictionary and lists
    temp_dict = {'Part#': [], 'Current': [], '+1': [], '+2': [], '+3': [], '+4': [], '+5': []}
    part_num_list = []
    current_list = []
    plus_one_list = []
    plus_two_list = []
    plus_three_list = []
    plus_four_list = []
    plus_five_list = []
    
    # Iterates over columns of temporary dataframe
    for c in range(fr_cols + 4, len(temp_df.columns)):
        
        # Get's date string from column names
        date_str = temp_df.columns[c][:-3]
        
        # Casts it to a datetime format
        adate = pd.to_datetime(date_str, format = '%Y-%m-%d')
        
        # Stores part number to list
        part_num_list.append(temp_df[temp_df.columns[2]][len(temp_df) - 1])
        
        # Iterates over rows of temporary detaframe
        for r in range(1, len(temp_df)):
            
            months_prior = 10
            
            # Stores first three characters of string
            string_start = temp_df[temp_df.columns[0]][r][:3]
            
            # If start of string matches condition
            if string_start == 'EDI':
            
                # Stores value from a given row
                row_value = temp_df[temp_df.columns[c]][r]
                
                # Stores date string from report name
                report_date_string = temp_df[temp_df.columns[0]][r][-10:]

                # Casts it to a datetime format
                report_date = pd.to_datetime(report_date_string, format = '%Y-%m-%d')

                # If date is greater than report date
                if adate >= report_date:

                    # Calculates with how much time in advance report was made
                    months_prior = 12 * (adate.year - report_date.year) + (adate.month - report_date.month)

            # If number of month's prior equals zero
            if months_prior == 0:

                # Adds value from given row to list
                current_list.append(row_value)

            # If number of month's prior equals one
            elif months_prior == 1:

                # Adds value from given row to list
                plus_one_list.append(row_value)

            # If number of month's prior equals two
            elif months_prior == 2:

                # Adds value from given row to list
                plus_two_list.append(row_value)

            # If number of month's prior equals three
            elif months_prior == 3:

                # Adds value from given row to list
                plus_three_list.append(row_value)

            # If number of month's prior equals four
            elif months_prior == 4:

                # Adds value from given row to list
                plus_four_list.append(row_value)

            # If number of month's prior equals five
            elif months_prior == 5:

                # Adds value from given row to list
                plus_five_list.append(row_value)
    
    # Making all lists the same length by adding zeros to the end of the shorter ones
    current_list.extend([0] * ((len(part_num_list)-len(current_list))))
    plus_one_list.extend([0] * ((len(part_num_list)-len(plus_one_list))))
    plus_two_list.extend([0] * ((len(part_num_list)-len(plus_two_list))))
    plus_three_list.extend([0] * ((len(part_num_list)-len(plus_three_list))))
    plus_four_list.extend([0] * ((len(part_num_list)-len(plus_four_list))))
    plus_five_list.extend([0] * ((len(part_num_list)-len(plus_five_list))))

    # Adds lists to dictionary
    temp_dict['Part#'] = part_num_list
    temp_dict['Current'] = current_list
    temp_dict['+1'] = plus_one_list
    temp_dict['+2'] = plus_two_list
    temp_dict['+3'] = plus_three_list
    temp_dict['+4'] = plus_four_list
    temp_dict['+5'] = plus_five_list
    
    # Creates a temporary dataframe using dictionary
    partial_df7 = pd.DataFrame(temp_dict)
    
    # Concatenates dataframes
    df7 = pd.concat([partial_df7, df7], ignore_index = True)

# Exporting to Excel.
df7.to_excel('Fill_rates_compiled.xlsx', float_format="%.2f")

# Creates empty dataframe
df8 = pd.DataFrame()

# Iterates over list of part numbers
for part_number in part_numbers:

    # Creates a temporary dataframe with first part number of the list
    temp_df1 = df7[df7[df7.columns[0]] == part_number]
    
    # Resets index of temporary dataframe
    temp_df1.reset_index(drop = True, inplace = True)
    
    # Replacing zeros with nan so that zeros don't affect mean values
    temp_df1.replace(0, np.nan, inplace=True)
    
    # Calculates the mean for each column of said dataframe
    temp_series = temp_df1[[temp_df1.columns[1], temp_df1.columns[2], temp_df1.columns[3], temp_df1.columns[4], temp_df1.columns[5],
                  temp_df1.columns[6]]].median(skipna = True)

    # Creates a temporary dictionary to hold means
    temp_dict = temp_series.to_dict()
    
    # Adds part number to the dictionary
    temp_dict['Part#'] = part_number

    # Creates a second temporary dataframe using the dictionary
    temp_df2 = pd.DataFrame(temp_dict, index = [0])

    # Concatenates df8 and the second temporary dataframe
    df8 = pd.concat([temp_df2, df8], ignore_index=True)
    
# Replacing nans with zeros
df8 = df8.fillna(0)

# Reorders columns
df8 = df8.iloc[:,[6, 0, 1, 2, 3, 4, 5]]

# Sorting by part number
df8.sort_values(df8.columns[0], inplace=True)

# Reseting index
df8.reset_index(drop = True, inplace = True)

# Exporting to Excel.
file_name = 'Median_fill_rates_' + str(current_date)[:-9] + '.xlsx'

df8.to_excel(file_name, float_format="%.2f")

# Stores end time and calculates time elapsed
end_time = time.time()
time_elapsed = end_time - start_time
print(f'Execution time was {time_elapsed}')
