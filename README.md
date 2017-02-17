# predict-AQI

Work in progress

- [x] Project proposal
- [x] Load & clean data
- [x] Explore Hypothesis # 1
- [x] Explore Hypothesis # 2
- [x] Explore Hypothesis # 3
- [x] Explore Hypothesis # 4
- [x] Produce final model & gather results
- [ ] Finish report

## Project Proposal

[Detailed proposal](./proposal.md)

This project is to make AQI predictions in the near future, given past AQI measurements of nearby locations (AQI is an integer measurement of air pollution). More specifically, the project aims to predict:

For a given location at a given point in time, what will the air pollution be every hour for the next 24 hours?

The baseline model to beat is one that predicts that the AQI for the next 24 hours will be the same AQI as right now.

Note for reviewer: [Proposal review](https://review.udacity.com/#!/reviews/267521)

## Project Report

[Detailed report](./report.md)

Four hypotheses were explored:

* [Time of day, day of week, day of month, and day of year can produce fairly accurate AQI predictions.](./predict_aqi/notebooks/hypothesis1_date_time.ipynb)
* [The recent AQI of a given location is a very good indicator of the near future AQI for that location.](./predict_aqi/notebooks/hypothesis2_recent_history.ipynb)
* [When combining recent AQI and time inputs to regressors to make a prediction, the number of input values needed for "recent AQI" is somewhere between six hours to one day.](./predict_aqi/notebooks/hypothesis3_history_depth.ipynb)
* [The AQI of nearby locations is a very good indicator of the near future AQI for a given location.](./predict_aqi/notebooks/hypothesis4_nearby_locations.ipynb)

The final model ended up being structured like this:

![original model](images/original_model.png)

Exploring these hypotheses gradually built up preprocessing, a pipeline, and a model. In the end, a hyperparameter grid search was performed and compared against the baseline model.

[Model Visualizations](./notebooks/final_model.ipynb)

No one set of hyperparameters for the model was optimal across all locations.

<insert top x model graphs>

Small improvements to the model and some changes to the testing process would be necessary to make the model production-ready (improvements link).

## Setup to Reproduce

Either use this docker setup for postgres / jupyter notebook or use your own setup of postgres / jupyter notebook and change the database config in `predict_aqi/load_data.py, line 7`.

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
* (For results of hyperparameter optimization) Download the results .csv files and put it in `predict_aqi/results_data/`: https://www.dropbox.com/sh/8vhx47k2xogtgct/AACL4eLQiB3cw7kezry3-fdSa?dl=0
* For the entire dataset (for postgres), download this xGB .zip file: 

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
