## Definition

### Project Overview x
Student provides a high-level overview of the project in layman’s terms. Background information such as the problem domain, the project origin, and related data sets or input data is given.

Air pollution was linked to 7 million premature deaths in 2012 [1] (~55 million people died in total in 2012 [2]). Using pollution masks when air pollution is bad can reduce premature deaths linked to air pollution. There are many consumer applications and services that broadcast what the air pollution is and has been, but there are very few that predict what air pollution will be. Similar to weather, individuals care more about air pollution in the near future, so they can make decisions about their day/week.

AQI (Air quality index) is a measurement that is commonly used to indicate air pollution. The number represents the greatest concentration (on a non-linear scale) of one of several types of harmful particles in the air. aqicn.org is a website that reports AQI in different locations around the world, and is used as a data source for this project.

### Problem Statement xx
The problem which needs to be solved is clearly defined. A strategy for solving the problem, including discussion of the expected solution, has been made.

For a given location at a given point in time, what will the air pollution be for the next 24 hours?

For a given city at a given point in time, the model will use the inputs:

* The recent AQI of the location
* The recent AQI of other nearby locations
* The current date and time

Transformed to produce the normalized features:

* The cos and sine of the minute of the day (cos and sine used to make the input circular - where 0 is 'close' to 1)
* The cos and sine of the day of the week
* The cos and sine of the day of the month
* The cos and sine of the day of the year
* The recent AQI of the location
* The recent AQI of other nearby locations

Used to predict the following output:

* An AQI prediction for every hour over the next 24 hours

The model I expect to build will look like the following:

![original model](images/original_model.png)



### Metrics xx
Metrics used to measure performance of a model or result are clearly defined. Metrics are justified based on the characteristics of the problem.

## Analysis

### Data Exploration xx
If a dataset is present, features and calculated statistics relevant to the problem have been reported and discussed, along with a sampling of the data. In lieu of a dataset, a thorough description of the input space or input data has been made. Abnormalities or characteristics about the data or input that need to be addressed have been identified.

### Exploratory Visualization xx
A visualization has been provided that summarizes or extracts a relevant characteristic or feature about the dataset or input data with thorough discussion. Visual cues are clearly defined.

### Algorithms and Techniques xx
Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.

### Benchmark xx
Student clearly defines a benchmark result or threshold for comparing performances of solutions obtained.

## Methodology

### Data Preprocessing xx
All preprocessing steps have been clearly documented. Abnormalities or characteristics about the data or input that needed to be addressed have been corrected. If no data preprocessing is necessary, it has been clearly justified.

### Implementation xx
The process for which metrics, algorithms, and techniques were implemented with the given datasets or input data has been thoroughly documented. Complications that occurred during the coding process are discussed.

### Refinement xx
The process of improving upon the algorithms and techniques used is clearly documented. Both the initial and final solutions are reported, along with intermediate solutions, if necessary.

## Results

### Model Evaluation and Validation xx
The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

### Justification xx
The final results are compared to the benchmark result or threshold with some type of statistical analysis. Justification is made as to whether the final model and solution is significant enough to have adequately solved the problem.

## Conclusion

### Free-form Visualization xx
A visualization has been provided that emphasizes an important quality about the project with thorough discussion. Visual cues are clearly defined.

### Reflection xx
Student adequately summarizes the end-to-end problem solution and discusses one or two particular aspects of the project they found interesting or difficult.

### Improvement xx
	
Discussion is made as to how one aspect of the implementation could be improved. Potential solutions resulting from these improvements are considered and compared/contrasted to the current solution.

## References xx



## Testing Hypotheses

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

The AQI of nearby locations is a very good indicator of the near future AQI for a given location. 


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
