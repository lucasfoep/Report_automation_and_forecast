library('tidyverse')
library('ggplot2') # Graphs
library('ggfortify') # Graphs
library('stats') # acf, pacf, box.test
library('forecast') # Arima, forecast, auto.arima and autoplot
library('lmtest') # coeftest
library('fUnitRoots') # adfTest
library('tseries') # garch
library(fGarch) # garchFit
library(rugarch) # ugarch
source('eacf.R')

# Loading dataframe and looking at it
channels <- read.csv('Channel_history_sales.csv')
head(channels)

# ================================================AFTERMARKET================================================

# Casting it to a time series object
aftermarket <- ts(channels$Aftermarket,
                           start = c(2020, 1),
                           frequency = 12)

# Looking at it
autoplot(aftermarket)
ggplot(aftermarket, aes(x=aftermarket)) + geom_histogram()
ggplot(aftermarket, aes(sample=aftermarket)) + stat_qq() + stat_qq_line()

autoplot(log(aftermarket))
ggplot(aftermarket, aes(x=log(aftermarket))) + geom_histogram()
ggplot(aftermarket, aes(sample=log(aftermarket))) + stat_qq() + stat_qq_line()

# Testing normality. The test fails to reject the null hypothesis of normality in both cases
# But the log distribution seems much closer to being normal
jarque.bera.test(aftermarket)
jarque.bera.test(log(aftermarket)) # Distribution looks much more normal.

# Very light downward trend and no seasonality
autoplot(decompose(aftermarket))

# A log transformation seems to help with normality
log.aftermarket <- log(aftermarket)

# Checking length of series
length(log.aftermarket)

# Splitting into train and test datasets
log.aftermarket.train <- subset(log.aftermarket, end = 28)
log.aftermarket.test <- subset(log.aftermarket, start = 29)

# This looks like an AR1
acf(log.aftermarket.train)
pacf(log.aftermarket.train)

# Trying a first model
m0 <- Arima(log.aftermarket.train, order = c(1,0,0))

# Both AR1 and intercept are very significant
coeftest(m0)

# Visually it seems ok.
fc0 <- forecast(m0, h = length(log.aftermarket.test))
plot(fc0)
lines(log.aftermarket.test, col = 'red')

# RMSFE of 47397.54
rmsfe.m0 <- sqrt(mean((exp(fc0$mean) - exp(log.aftermarket.test))^2))
rmsfe.m0

# About 26.7%.
mape.m0 <- mean(abs((exp(fc0$mean) - exp(log.aftermarket.test))/exp(log.aftermarket.test)))
mape.m0

# Storing results
fc0.results <- exp(fc0$mean)
fc0.lower <- exp(fc0$lower)
fc0.upper <- exp(fc0$upper)

# ===============================================Intercompany.Internatio=================================

channels$Intercompany.Internatio[channels$Intercompany.Internatio < 0] <- 0

# Casting it to a time series object
interco <- ts(channels$Intercompany.Internatio,
                  start = c(2020, 1),
                  frequency = 12)

# Looking at it
autoplot(interco)
ggplot(interco, aes(x=interco)) + geom_histogram()
ggplot(interco, aes(sample=interco)) + stat_qq() + stat_qq_line()

# Looks "normal enough"
jarque.bera.test(interco)

# Very light downward trend and no seasonality
autoplot(decompose(interco))

# Checking length of series
length(interco)

# Splitting into train and test datasets
interco.train <- subset(interco, end = 28)
interco.test <- subset(interco, start = 29)

# This looks like an MA1
acf(interco.train)
pacf(interco.train)

# Trying a first model
m1 <- Arima(interco.train, order = c(0,0,1))

# Only intercept seems to be significant
coeftest(m1)

# Definitely not great.
fc1 <- forecast(m1, h = length(interco.test))
plot(fc1)
lines(interco.test, col = 'red')

# RMSFE of 55053.33
rmsfe.m1 <- sqrt(mean((fc1$mean - interco.test)^2))
rmsfe.m1

# HORRIBLE.
mape.m1 <- mean(abs((fc1$mean - (interco.test + 1))/(interco.test+1)))
mape.m1

# Storing results
fc1.results <- fc1$mean
fc1.lower <- fc1$lower
fc1.upper <- fc1$upper

# ===============================================OE===============================================

# Casting it to a time series object
oe <- ts(channels$OE,
                  start = c(2020, 1),
                  frequency = 12)

# Looking at it
autoplot(oe)
ggplot(oe, aes(x=oe)) + geom_histogram()
ggplot(oe, aes(sample=oe)) + stat_qq() + stat_qq_line()

# Distribution seems normal
jarque.bera.test(oe)

# Very light upward trend and no seasonality
autoplot(decompose(oe))

# Checking length of series
length(oe)

# Splitting into train and test datasets
oe.train <- subset(oe, end = 28)
oe.test <- subset(oe, start = 29)

# This looks like an AR1
acf(oe.train)
pacf(oe.train)

# Trying a first model
m2 <- Arima(oe.train, order = c(0,1,1), include.drift = TRUE)

# MA is very significant, drift is still significant
coeftest(m2)

# Visually it seems ok.
fc2 <- forecast(m2, h = length(oe.test))
plot(fc2)
lines(oe.test, col = 'red')

# RMSFE of 607426.1
rmsfe.m2 <- sqrt(mean((fc2$mean - oe.test)^2))
rmsfe.m2

# About 9%.
mape.m2 <- mean(abs((fc2$mean - oe.test)/oe.test))
mape.m2

# Storing results
fc2.results <- fc2$mean
fc2.lower <- fc2$lower
fc2.upper <- fc2$upper


# ===============================================OE.MEXICO==============================================

# Casting it to a time series object
oe.mex <- ts(channels$OE.MEXICO,
         start = c(2020, 1),
         frequency = 12)

# Looking at it
autoplot(oe.mex)
ggplot(oe.mex, aes(x=oe.mex)) + geom_histogram()
ggplot(oe.mex, aes(sample=oe.mex)) + stat_qq() + stat_qq_line()

# Distribution seems normal
jarque.bera.test(oe.mex)

# No trend or seasonality
autoplot(decompose(oe.mex))

# Checking length of series
length(oe.mex)

# Splitting into train and test datasets
oe.mex.train <- subset(oe.mex, end = 28)
oe.mex.test <- subset(oe.mex, start = 29)

# This looks like an MA1
acf(oe.mex.train)
pacf(oe.mex.train)

# Trying a first model
m3 <- Arima(oe.mex.train, order = c(0,1,1))

# MA is very significant
coeftest(m3)

# Visually it seems ok.
fc3 <- forecast(m3, h = length(oe.mex.test))
plot(fc3)
lines(oe.mex.test, col = 'red')

# RMSFE of 289998
rmsfe.m3 <- sqrt(mean((fc3$mean - oe.mex.test)^2))
rmsfe.m3

# About 9%.
mape.m3 <- mean(abs((fc3$mean - oe.mex.test)/oe.mex.test))
mape.m3

# Storing results
fc3.results <- fc3$mean
fc3.lower <- fc3$lower
fc3.upper <- fc3$upper


# ===============================================OES===============================================

# Casting it to a time series object
oes <- ts(channels$OES,
             start = c(2020, 1),
             frequency = 12)

# Looking at it
autoplot(oes)
ggplot(oes, aes(x=oes)) + geom_histogram()
ggplot(oes, aes(sample=oes)) + stat_qq() + stat_qq_line()

autoplot(log(oes))
ggplot(log(oes), aes(x=log(oes))) + geom_histogram()
ggplot(log(oes), aes(sample=log(oes))) + stat_qq() + stat_qq_line()

# Distribution seems normal with a log transformation
jarque.bera.test(oes)
jarque.bera.test(log(oes))

# Storing log for the series
log.oes <- log(oes)

# No trend or seasonality
autoplot(decompose(log.oes))

# Checking length of series
length(log.oes)

# Splitting into train and test datasets
log.oes.train <- subset(log.oes, end = 28)
log.oes.test <- subset(log.oes, start = 29)

# AR1? MA2? ARMA(1,1)?
acf(log.oes.train)
pacf(log.oes.train)

# Trying a first model
m4 <- Arima(log.oes.train, order = c(1,0,0))

# AR1 is significant and intercept is very significant
coeftest(m4)

# Visually it seems ok.
fc4 <- forecast(m4, h = length(log.oes.test))
plot(fc4)
lines(log.oes.test, col = 'red')

# RMSFE of 188427.2
rmsfe.m4 <- sqrt(mean((exp(fc4$mean) - exp(log.oes.test))^2))
rmsfe.m4

# About 15%.
mape.m4 <- mean(abs((exp(fc4$mean) - exp(log.oes.test))/exp(log.oes.test)))
mape.m4

# Storing results
fc4.results <- exp(fc4$mean)
fc4.lower <- exp(fc4$lower)
fc4.upper <- exp(fc4$upper)

# ===============================================OESMEXICO===============================================

channels$OES.MEXICO[channels$OES.MEXICO < 0] <- 0

# Casting it to a time series object
oes.mex <- ts(channels$OES.MEXICO,
          start = c(2020, 1),
          frequency = 12)

# Looking at it
autoplot(oes.mex)
ggplot(oes.mex, aes(x=oes.mex)) + geom_histogram()
ggplot(oes.mex, aes(sample=oes.mex)) + stat_qq() + stat_qq_line()

autoplot(log(oes.mex + 1))
ggplot(log(oes.mex + 1), aes(x=log(oes.mex + 1))) + geom_histogram()
ggplot(log(oes.mex + 1), aes(sample=log(oes.mex + 1))) + stat_qq() + stat_qq_line()

# Distribution seems normal with a log transformation
jarque.bera.test(oes.mex)
jarque.bera.test(log(oes.mex + 1))

iqr <- IQR(log(oes.mex + 1))
quantile(log(oes.mex + 1))
q1 <- 6.941441
q3 <- 9.923376
LL <- q1 - iqr * 1.5
UL <- q3 + iqr * 1.5

# Storing log for the series
log.oes.mex <- log(oes.mex + 1)

log.oes.mex <- log.oes.mex[(log.oes.mex > LL) & (log.oes.mex < UL)]

log.oes.mex <- ts(log.oes.mex,
              start = c(2020, 1),
              frequency = 12)

# Looking at it
autoplot(log.oes.mex)
ggplot(log.oes.mex, aes(x=log.oes.mex)) + geom_histogram()
ggplot(log.oes.mex, aes(sample=log.oes.mex)) + stat_qq() + stat_qq_line()

jarque.bera.test(log.oes.mex) # Still not a normal distribution


# ===============================================Rings===============================================


# Casting it to a time series object
rings <- ts(channels$Rings.300,
              start = c(2020, 1),
              frequency = 12)

# Looking at it
autoplot(rings)
ggplot(rings, aes(x=rings)) + geom_histogram()
ggplot(rings, aes(sample=rings)) + stat_qq() + stat_qq_line()

# Distribution seems normal enough
jarque.bera.test(rings)

# No trend or seasonality
autoplot(decompose(rings))

# Checking length of series
length(rings)

# Splitting into train and test datasets
rings.train <- subset(rings, end = 28)
rings.test <- subset(rings, start = 29)

# MA1?
acf(rings.train)
pacf(rings.train)

# Trying a first model
m5 <- Arima(rings.train, order = c(0,0,1))

# MA1 and intercept are both very significant
coeftest(m5)

# Visually it seems ok.
fc5 <- forecast(m5, h = length(rings.test))
plot(fc5)
lines(rings.test, col = 'red')

# RMSFE of 127141.8
rmsfe.m5 <- sqrt(mean((fc5$mean - rings.test)^2))
rmsfe.m5

# About 54%.
mape.m5 <- mean((abs(fc5$mean) - rings.test)/rings.test)
mape.m5

# Storing results
fc5.results <- fc5$mean
fc5.lower <- fc5$lower
fc5.upper <- fc5$upper

# ===============================================Final models===============================================
# Aftermarket
m0f <- Arima(log.aftermarket, order = c(1,0,0))
fc0f <- forecast(m0f, h = 4)
plot(fc0f)
fc0f.results <- exp(fc0f$mean)
fc0f.lower <- exp(fc0f$lower)
fc0f.upper <- exp(fc0f$upper)

# International
m1f <- Arima(interco, order = c(0,0,1))
fc1f <- forecast(m1f, h = 4)
plot(fc1f)
fc1f.results <- fc1f$mean
fc1f.lower <- fc1f$lower
fc1f.upper <- fc1f$upper

# OE
m2f <- Arima(oe, order = c(0,1,1), include.drift = TRUE)
fc2f <- forecast(m2f, h = 4)
plot(fc2f)
fc2f.results <- fc2f$mean
fc2f.lower <- fc2f$lower
fc2f.upper <- fc2f$upper

# OE MEX
m3f <- Arima(oe.mex, order = c(0,1,1))
fc3f <- forecast(m3f, h = 4)
plot(fc3f)
fc3f.results <- fc3f$mean
fc3f.lower <- fc3f$lower
fc3f.upper <- fc3f$upper

# OES
m4f <- Arima(log.oes, order = c(1,0,0))
fc4f <- forecast(m4f, h = 4)
plot(fc4f)
fc4f.results <- exp(fc4f$mean)
fc4f.lower <- exp(fc4f$lower)
fc4f.upper <- exp(fc4f$upper)

# Rings
m5f <- Arima(rings, order = c(0,0,1))
fc5f <- forecast(m5f, h = 4)
plot(fc5f)
fc5f.results <- fc5f$mean
fc5f.lower <- fc5f$lower
fc5f.upper <- fc5f$upper

