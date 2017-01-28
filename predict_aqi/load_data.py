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
    ).where(
        AirMeasurement.measurement_datetime != None
    ).order_by(AirMeasurement.id.asc())
    return pd.read_sql(s, engine)


def wkt_string_to_lat_lon(wkt_string):
    if not wkt_string:
        return 0, 0
    return [float(x) for x in wkt_string.split('(')[1].split(')')[0].split(' ')]


def load_nearby_locations(airlocation_id, distance_km):
    '''
    Return a list of airlocation details within `distance_km` of location with id `airlocation_id`
    '''
    airlocation_wkt_string = execute_raw_sql(
        "SELECT ST_ASEWkt(coordinates) from locs_airlocation "
        "WHERE id={}".format(airlocation_id)
    ).scalar()
    lon, lat = wkt_string_to_lat_lon(airlocation_wkt_string)
    lat_lon_point_string = "(ST_MakePoint({}, {}))".format(lon, lat)
    coordinate_point_string = "ST_Point(ST_X(ST_Centroid(coordinates)), ST_Y(ST_Centroid(coordinates)))"
    location_data = execute_raw_sql(
        "SELECT id, ST_Distance({coordinate_point_string}, {lat_lon_point_string}) FROM locs_airlocation "
        "WHERE GeometryType(ST_Centroid(coordinates)) = 'POINT' AND "
        "ST_Distance_Sphere({coordinate_point_string}, {lat_lon_point_string}) "
        "<= {distance_km} * 2589.981673".format(
            **locals()
        )
    )
    return sorted(list(location_data), key=lambda a: a[1])


def execute_raw_sql(sql_string):
    return session.execute(sql_string)


def load_nearby_airlocations_to_list_of_dataframes(airlocation_id, distance_km, max_locations=3, print_progress=False):
    nearby_locations = load_nearby_locations(airlocation_id, distance_km)
    if print_progress:
        print("Predicting for: " + load_air_location_data(nearby_locations[0][0])['en_city_name'])
        print("Using nearby locations: " + str([load_air_location_data(l[0])["short_name"] for l in nearby_locations[1:]]))

    loc_columns_format = ['loc_{}_id', 'loc_{}_measurement_datetime', 'loc_{}_aqi']

    def initial_load_and_format_airlocation(airlocation_id, index):
        single_loc_df = load_measurement_data(airlocation_id)
        single_loc_df.columns = [c.format(index) for c in loc_columns_format]
        if print_progress:
            print("{} rows loaded for loc #{}".format(single_loc_df.count()[0], airlocation_id))
        return single_loc_df

    airlocation_ids = [l[0] for l in nearby_locations[:max_locations]]

    df_list = [initial_load_and_format_airlocation(nearby_locations[0][0], 1)]

    for index, id in enumerate(airlocation_ids[1:max_locations]):
        loc_df = initial_load_and_format_airlocation(id, index + 2)
        df_list.append(loc_df)

    return df_list, airlocation_ids
