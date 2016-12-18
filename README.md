# predict-AQI

## Project Proposal

[Detailed proposal](./proposal.md)

AQI is an integer measurement of air pollution. This project is to make an AQI prediction in the near future, given past AQI measurements of nearby locations. The project aims to predict:

For a given city at a given point in time, what will the air pollution be for the next 24 hours?

The baseline model to beat is one that predicts that the next 24 hours will be the same AQI as right now.

## Project Report

[Detailed report](./report.md)

Four hypotheses were explored:

* [Time of day, day of week, day of month, and day of year can produce fairly accurate AQI predictions.](./notebooks/hypothesis1_date_time.ipynb)
* [The recent AQI of a given location is a very good indicator of the near future AQI for that location.](./notebooks/hypothesis2_recent_history.ipynb)
* [When combining recent AQI and time inputs to regressors to make a prediction, the number of input values needed for "recent AQI" is somewhere between six hours to one day.](./notebooks/hypothesis3_history_depth.ipynb)
* [The AQI of nearby locations is a very good indicator of the near future AQI for a given location.](./notebooks/hypothesis4_nearby_locations.ipynb)

If all the hypotheses were correct, the final model would have ended up looking something like:

*** Insert structural picture here ***

However, one hypothesis proved incorrect, so the final model ended up looking more like:

*** Insert structural picture here ***

Exploring these hypotheses gradually built up a pipeline and model. In the end, a final predictive model that was built makes a prediction on the next 24 hours of AQI for a location, given the AQI history of nearby locations and current point in time.

[Model Visualizations](./notebooks/final_model.ipynb)

## Setup to Reproduce

Either use this docker setup for postgres / jupyter notebook or use your own setup of postgres / jupyter notebook and change the database config in `load_data.py`.

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
* Download the data set
  * *** LINK TO DATA SET ***
* Load the database from the dump file to the postgres db:
```
commands to do that
```
* Use an existing notebook, (***insert final model notebook here***) recommended

## Libraries Used

For data transformation and prediction:
* Python
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