### Domain Background

Air pollution was linked to 7 million premature deaths in 2012 [1], and ~55 million people died in 2012 [2]. Using pollution masks when air pollution is bad can reduce premature deaths linked to air pollution. There are many consumer applications and services that broadcast what the air pollution is and has been, but there are very few that predict what air pollution will be. Similar to weather, individuals care more about air pollution in the near future, so they can make decisions about their day/week.

AQI (Air quality index) is a measurement that is commonly used to indicate air pollution. The number represents the greatest concentration (on a non-linear scale) of one of several types of harmful particles in the air. [aqicn.org](http://aqicn.org/) is a website that reports AQI in different locations around the world, and is used as a data source for this project.

The factors that influence air quality can be very complex - air quality depends on human and naturally-occurring pollution entering the atmosphere in various states, then being further dispersed into the atmosphere [3]. Models for weather forecasting can be used in air quality forecasting [4]. In my approach to any prediction problem in this domain, I will have to reduce the scope of the modelling and inputs to make a solution possible.

### Problem Statement

For a given city at a given point in time, what will the air pollution be for the:

* next 24 hours? (required)

* next week? (less important, nice-to-have)

### Datasets and Inputs

The dataset is the AQI data for some locations between September 2015 - September 2016. The data contains the AQI of each location at 30-minute intervals. The ~3000 locations contained in the data set are mostly ones in China, India, Japan, and Canada, and mostly locations within cities. The latitudes and longitudes of the locations are known. The data was publicly available, obtained from [aqicn.org](http://aqicn.org/), which collected the data from various government agencies and networks of shared AQI sensor machines.

Measurements are likely made by very different machines across the world and across countries. From anecdotal experience, AQI levels for different sensors (with different locations) in the same city are still reasonably close (within 10 units). The date and time are all in the same timezone (so the time is not necessarily the local time of day).

A single measurement data point contains:

* The integer AQI level

* The time and date (datetime) when the measurement was collected

* The location of the measurement (country, city, latitude, longitude)

The postgresql dump of the dataset is ~16gB in size. For the purposes of grading/reviewing this project I will reduce the number of locations in the data set so the dump is ~5gB in size. Some data points are explicitly invalid, indicated by a 0 or -1. Some data points are outliers and obviously wrong - ex, when a sequence of readings every 30min for a given location looks like: 72, 69, 70, 687, 72.

### Solution Statement

For a given city at a given point in time, the model will use the inputs:

* The recent AQI history of the location (on the order of 1 to 10 days)

* The location’s latitude and longitude

* The recent AQI history of other locations (and their latitudes/longitudes)

* The current date and time

* The long-term AQI history of the location (possibly)

To produce the output:

* An AQI prediction for every hour over the next 24 hours for this location (required)

* An AQI prediction for every hour over the next week (less important, nice-to-have)

To train, the model will perform the following preprocessing:

* Scale the AQI numbers from 0 to 1 - this will require exploring the larger AQI number in the dataset to find an upper-bound number that enables high accuracy without excluding high-AQI predictions

* Convert the latitudes and longitudes of nearby cities into the distance from the target city’s latitude and longitude, so the distance can be used as a way to scale the importance of each location’s recent AQI history. The distances will be scaled to be within 0-1, where 1 is the largest distance between "nearby cities" used in the model

* Change the current date and time into:

    * The current hour of day

    * The current day of week

    * The current day of month

    * The current day of year

* For all AQI numbers that are invalid (described in Dataset and Inputs), treat them as if their value is the last valid AQI number for the city, but don’t learn or train on them (they should not be used as a target AQI to guess)

Then the model will split the data into a test/training set (see Evaluation Metrics). Finally, the model will train on the training set:

1. Train recurrent neural networks for each location.

2. Train neural network combining nearby locations’ prediction data with current date and time for each location.

### Benchmark Model

In this problem, we will have the "correct" result of every prediction since the correct answer to the prediction will eventually become input data for future predictions.

A simple model to benchmark against is a model that predicts all future AQI measurements to be the current AQI measurement for a given location. The solution must perform better than this model.

### Evaluation Metrics

Since the time of year can be relevant to the prediction, and the data set is limited, it’s difficult to have an unbiased test sample. For the test set, random one-day periods are chosen throughout the year (and removed from the training set). A list of random points in time (where the points of time take place at the beginning of the one-day period) will be chosen.

For each point in time, t0, the AQI at 24 points in time in one hour intervals starting an hour later will have to be predicted, t1, …, t24 . The input to the predictor is all of the data points that would have been available to the system at t0 to predict the AQI at t1, …, t24. Namely, all data points with time at or earlier than t0.

The model must minimize the mean squared difference between predicted AQI and actual AQI for t1, …, t24 of the test set. For a consumer, being ~10 AQI off 10 times is better than being ~100 AQI off once, so mean squared error is more applicable than mean absolute error.

### Project Design

For this problem, I have the following hypotheses:

1. The time of year and time of day both influence the near future AQI of a location.

2. Different locations have different patterns.

3. Recent AQI of a location heavily influences the near future AQI of a location.

4. The distant past is a poor predictor of the near future AQI of a location.

5. The recent AQI of nearby locations influence the near future AQI of a location.

My workflow will be to explore these hypotheses and incorporate them into a model. I will:

1. Load & clean the data from postgresql (described in Solution Statement)

2. Explore hypothesis #1 (time influences prediction).

    1. Turn the date and time of each measurement into inputs to a SVM regressor:

        1. Minute of the day (evening usually has less pollution than daytime)

        2. Day of the year (seasonal changes)

        3. Day of the week (weekday vs. weekend influences)

        4. Day of the month (unlikely to influence)

    2. Visualize the error rate and compare against the benchmark model.

3. Explore hypothesis #2 (different locations have different patterns).

    3. Train the model on one location at a time and compare the predictive power against the model that ignores locational differences.

4. Explore hypothesis #3 (recent history is a strong predictor).

    4. Use a single location’s time series data - try to predict the next few integers, given a series of integers using a recurrent neural network. 

    5. Visualize predictions against results and compare error to the baseline model.

    6. Try other locations and visualize the results in a similar manner.

5. Explore hypothesis #4 (distant past is a poor predictor)

    7. Find the time delta into the past where the data is no longer useful to the recurrent neural network as a predictor (while still using the time of day/year as inputs).

6. Explore hypothesis #5 (nearby locations’ history is a strong predictor).

    8. For a location, use the time series data of all the locations within a radius (and their distances) to make predictions for each location. Use each location’s predictions as an input to a neural network, using the distances to influence initial weights.

    9. Visualize the predictions and compare to just the location’s predictions.

    10. Try different radiuses.

    11. Try different locations.

7. Add the time and date data as inputs to the neural network in #6.

Since part of the problem is time-series forecasting regression, multilayer perceptron and Gaussian process regression are good candidates for modeling [5]. Strategies for time-series forecasting are MIMO, direct, and H-step-ahead [6]

Since the data is locational, there are likely more advanced modelling techniques I could use. I will start by simply using distance from the target location (along with the time series data) as an input.

Since I’m combining different types of inputs, I will have to ensemble the models together.

[1] World Health Organization, March 2014, *7 million premature deaths annually linked to air pollution*, [http://www.who.int/mediacentre/news/releases/2014/air-pollution/en/](http://www.who.int/mediacentre/news/releases/2014/air-pollution/en/) 

[2] The World Bank, October 2016, *Death rate, crude (per 1,000 people)*, [http://data.worldbank.org/indicator/SP.DYN.CDRT.IN](http://data.worldbank.org/indicator/SP.DYN.CDRT.IN)

[3] Wikipedia, October 2016, *Atmospheric dispersion modeling*, [https://en.wikipedia.org/wiki/Atmospheric_dispersion_modeling](https://en.wikipedia.org/wiki/Atmospheric_dispersion_modeling) 

[4] J. Michalakes, J. Dudhia, D. Gill, T. Henderson, J. Klemp, W. Skamarock, W. Wang, 2004, *The Weather Research And Forecast Model: Software Architecture And Performance*, 

[http://www.wrf-model.org/wrfadmin/docs/ecmwf_2004.pdf](http://www.wrf-model.org/wrfadmin/docs/ecmwf_2004.pdf) 

[5] Nesreen K. Ahmed , Amir F. Atiya , Neamat El Gayar , Hisham El-shishiny, September 2010, *An Empirical Comparison of Machine Learning Models for Time Series*

*Forecasting*

[6] Gianluca Bontempi, 2013, *Machine Learning Strategies for Time*

*Series Prediction*, [http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf](http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf) 

