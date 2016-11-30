from sqlalchemy import Column, Integer, MetaData, Table, create_engine, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.sql import select
import pandas as pd

engine = create_engine('postgres://douglas@postgis:5432/AQI', client_encoding='utf8')
Session = sessionmaker(bind=engine)
session = Session()

metadata = MetaData()
metadata.reflect(engine, only=['locs_airlocation', 'locs_airmeasurement'])
Base = automap_base(metadata=metadata)
Base.prepare()

AirLocation, AirMeasurement = Base.classes.locs_airlocation, Base.classes.locs_airmeasurement


def get_db_session():
    return session


def load_air_location_data(airlocation_id):
    '''
    Return the air location data object for id `airlocation_id`
    '''
    s = select([AirLocation]).where(AirLocation.id == airlocation_id)
    return session.execute(s).first()


def load_measurement_data(airlocation_id):
    '''
    Return a dataframe of the air quality measurement data for a location where the AQI is a valid number.
    '''
    s = select([AirMeasurement.id, AirMeasurement.measurement_datetime, AirMeasurement.aqi]).where(
        AirMeasurement.airlocation_id == airlocation_id
    ).where(
        AirMeasurement.aqi != None
    ).where(
        AirMeasurement.aqi != 0
    ).order_by(AirMeasurement.id.asc())
    return pd.read_sql(s, engine)


def load_nearby_location_measurement_data(airlocation_id, distance_km, max_cities=10):
    '''
    Return a list of dataframes of location air quality measurement data within `distance_km` of location with id
    `airlocation_id`
    '''
    pass


