# Automated forecast

### What does it do?

This project automates a forecasting report for the demand of over 700 customer products by reading over 30 excel files, cleaning the data, merging dataframes, applying formulas that are standard practice in their industry and outputting a few summarized reports. The customer has access to a service that supplies them with how much their customers plan on manufacturing of each one of their products in the following months. The first step was to calculate a "fill rate", i.e their customers' historical average percentage of the amount planned on being manufactured per product that actually became firm orders. The subsequent step was to merge spreadsheets containing information such as orders booked for the month, planned orders and historical sales with the calculated fill rates, to then apply the formulas to forecast demand for the customer's products.

### Improvements being made

In lieu of third party information that would allow for the use of the industry standard formulas, the customer determined that a six months average was to be applied. Throughout the development of this project it has been found that such method is not accurate, let alone for long range forecasts, demanding considerable manual input based on the customer's expertise. In the light of that find, time series forecasting models are being developed. However, the data for the products for which there is a lack of said third party information is sparse. Those products are also a majority and using time series forecasting on each one of them would be time consuming. In the interest of delivering a usable product in a timely manner, preliminary models have been built to forecast products grouped into channels. The next step is to fill in the gaps to allow for time series forecasting for each individual product.

### What were the tools used?

- Pandas
- Numpy
- Matplotlib
