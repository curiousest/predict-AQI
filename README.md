# predict-AQI

## Project Proposal

[Detailed proposal](./proposal.md)

This project is to make AQI predictions in the near future, given past AQI measurements of nearby locations (AQI is an integer measurement of air pollution). More specifically, the project aims to predict:

For a given location at a given point in time, what will the air pollution be every hour for the next 24 hours?

The baseline model to beat is one that predicts that the AQI for the next 24 hours will be the same AQI as right now.

## Project Report

[Detailed report](./report.md)

[pdf](./images/report.pdf)

Four hypotheses were explored:

* [Time of day, day of week, day of month, and day of year can produce fairly accurate AQI predictions.](./predict_aqi/notebooks/hypothesis1_date_time.ipynb)
* [The recent AQI of a given location is a very good indicator of the near future AQI for that location.](./predict_aqi/notebooks/hypothesis2_recent_history.ipynb)
* [When combining recent AQI and time inputs to regressors to make a prediction, the number of input values needed for "recent AQI" is somewhere between six hours to one day.](./predict_aqi/notebooks/hypothesis3_history_depth.ipynb)
* [The AQI of nearby locations is a very good indicator of the near future AQI for a given location.](./predict_aqi/notebooks/hypothesis4_nearby_locations.ipynb)

The final model ended up being structured like this:

![original model](images/original_model.png)

Exploring these hypotheses gradually built up preprocessing, a pipeline, and a model. In the end, a hyperparameter grid search was performed and compared against the baseline model.

[Model Visualizations](./predict_aqi/notebooks/final_model.ipynb)

No one set of hyperparameters for the model was optimal across all locations. That being said, different hyperparameters for the model performed better than the baseline model.

These are the results for 50 different sets of hyperparameters on a single location compared to the baseline model. The x-axis represents predictions n hours in the future. The y-axis represents the average absolute error for the predictor. That means a single point is the average absolute error for a given predictor predicting n hours ahead for a given location. The red line is the baseline model error.

![Error on top 50 models](images/error_top_50s.png)

Small improvements to the model and some changes to the testing process would be necessary to make the model production-ready (see [improvements](./report.md#improvement)).

## Setup to Reproduce

Either use this docker setup for postgres / jupyter notebook or use your own setup of postgres / jupyter notebook and change the database config in `predict_aqi/load_data.py, line 7`.

### If you want to use docker

* Install docker and docker-compose (this can be non-trivial, especially for mac/windows):
  * https://docs.docker.com/engine/installation/
  * https://docs.docker.com/compose/install/
* Check that your installation worked:
  * Make sure that the ports used by postgres / jupyter notebook are not in use (by turning off the applications that would use them)
  * Then:
```bash
$ cd predict-AQI
$ docker-compose up -d
Starting predictaqi_postgis_1
Starting predictaqi_notebook_1
```

### Retrieving the data

* For results of hyperparameter optimization, download the results .csv files and put it in `predict_aqi/results_data/`: https://www.dropbox.com/sh/8vhx47k2xogtgct/AACL4eLQiB3cw7kezry3-fdSa?dl=0
* For a third of the entire dataset, download this 1.8GB .sql file (for postgres): https://dl.dropboxusercontent.com/u/22651169/half_data_dump.sql

## Libraries Used

For data transformation and prediction:
* Python 3
* NumPy
* Pandas
* Matplotlib
* Scikit Learn

For storing / querying data:
* Postgres
* Postgres GIS
* SQLAlchemy

For running everything:
* Docker
* Docker-compose
* Jupyter Notebook (jupyter/datascience-notebook)
