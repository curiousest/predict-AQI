## Hypothesis

### Hypothesis 1

Time of day, day of week, day of month, and day of year are fairly strong predictors of AQI.

Had to turn data into circular inputs. Interpolation >> Extrapolation, obviously. Interpolation used as sanity check.

### Hypothesis 2

The recent AQI of a given location is a very good indicator of the near future AQI for that location.

### Hypothesis 3

When combining recent AQI and time regressors to make a prediction, the number of values needed for "recent AQI" isn't worth having much more than a day.
 
 * Build model to train against first 90% of data on both time and recent AQI
 * Combine time and recent AQI predictions with neural network
 * Try turning the time outputs into one for each, then combine with the recent AQI prediction in neural network (5 inputs total to neural network per-datetime)
 * Paramaterize how distant to look into the past and graph the means squared error of AQI (not normalized AQI)

### Hypothesis 4

The recent AQI of nearby locations is a very good indicator of the near future AQI for a given location. 

### Hypothesis 5

A predictor (using recent AQI history + time) trained across all locations will not add predictive power to one trained on nearby locations (using recent AQI history + time).


### Tune the predictor

 * Try on different cities
 * Do the multi-step testing (train on 1st month, predict second; train on 1-2mo test on 3rd; train on 1-3mo test on 4th, etc.)
 * Find how far to look into the past for nearby cities
 * Different regressor for time predictor
 * Recurrent neural networks instead of normal for recent AQI
 * 
 * Clean the data better? (use actual 30 min intervals)

## Notes

### Problem Statement

 * Dataset is actually ~5gB, not 16gB
 * Lots of the data is just null values
 * Benchmark model is actually predicting the same AQI for the next 24 hours
 * The data isn't in perfect 30min intervals

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
