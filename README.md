# Automated forecast

### What it does

This project automates a forecasting report for the demand of over 700 customer products by reading over 30 excel files, cleaning the data, merging dataframes, applying formulas that are standard practice in their industry and outputting a few summarized reports. The customer has access to a service that supplies them with how much their customers plan on manufacturing of each one of their products in the following months. The first step was to calculate a "fill rate", i.e their customers' historical percentage average of the amount planned on being manufactured per product that became firm orders. The subsequent step was to merge spreadsheets containing information such as orders booked for the month, planned orders and historical sales with the calculated fill rates, to then apply the formulas to forecast demand for the customer's products.

### Challenges

The data used had been exported to excel sheets rather than extracted directly from the database. The excel sheets reports used as input stored data either on a monthly or weekly basis, requiring extensive use of the datetime library as well as finding ways of combining different reports. On a weekly level, split weeks had to be considered so as to allocate values on the correct months. Products' expected demands were grouped by product rather than kept discriminated by customer, and a mechanism to re-distribute it by customer had to be designed.

### Improvements being made

In lieu of expected demand, which would allow for the use of the industry standard formulas, the customer determined that a six months average was to be applied. Throughout the development of this project it has been found that such method is not accurate, let alone for long range forecasts, demanding considerable manual input based on the customer's expertise. In the light of that find, time series forecasting models are being developed. However, the historical sales data for the products for which there is a lack of expected demand is sparse. Those products are also the majority and using time series forecasting on each one of them would be time consuming. In the interest of delivering a usable product in a timely manner, preliminary models have been built to forecast products grouped into channels. The next step is to fill in the gaps to allow for time series forecasting for each individual product and start building time series forecasting models for each individual project.

### Tools used

- Python: Pandas, Numpy, Glob, Os, Time, Calendar, Math, Datetime.

- R: ggplot2, stats, forecast, lmtest, fUnitRoots.

### PS

In the interest of maintaining customer's confidentiality the script was published as a python file rather than a jupyter notebook. On a Lenovo Legion pro 7, 13th i9 2.2GHz, 32 GB RAM, nvidia geforce rtx 4080 Fill_rates.py runs in about 30s (once a month); Forecast.py runs in about 20s (weekly) and Channels_forecast.R runs in about 10s (weekly).

