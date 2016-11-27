import math
import itertools
from predict_aqi import config


def make_circular_input(input, input_max, input_min=0):
    '''
    Convert the input to a circular input between 0-1.
    Circular input being two values which cause the smallest and
    largest values inputted to this function appear to be beside each other.

    This is used, for example, to make the last minute of the day "close" to
    the first minute of the day.
    '''
    # normalize the input to a real 0-1, then project it along
    # the first period of the sine/cosine waves
    normalized_projected = 2 * math.pi * (input - input_min) / (input_max - input_min)

    # apply sine / cosine and convert to a real number 0-1
    sine = (1 + math.sin(normalized_projected)) / 2
    cosine = (1 + math.cos(normalized_projected)) / 2
    assert 1 >= sine >= 0
    assert 1 >= cosine >= 0
    return sine, cosine


def get_normalized_time_inputs(dt):
    return tuple(itertools.chain(
        # minute of day
        make_circular_input((dt.time().hour * 60) + dt.time().minute, 1440),
        # day of the year
        make_circular_input(dt.timetuple().tm_yday, 365),
        # day of the week, from 0 to 6
        make_circular_input(dt.date().weekday(), 6),
        # day of the month
        make_circular_input(dt.date().day, 31, 1),
    ))


def generate_time_inputs(source_df, dest_df):
    dest_df['minute_of_day_sin'], dest_df['minute_of_day_cos'], \
    dest_df['day_of_year_sin'], dest_df['day_of_year_cos'], \
    dest_df['day_of_week_sin'], dest_df['day_of_week_cos'], \
    dest_df['day_of_month_sin'], dest_df['day_of_month_cos'] = \
        zip(*source_df["measurement_datetime"].map(get_normalized_time_inputs))


def get_normalized_aqi(aqi):
    if not aqi:
        return 0
    return min(aqi, config.MAX_AQI) / 300


def get_denormalized_aqi(normalized_aqi):
    try:
        if normalized_aqi is None or normalized_aqi <= 0:
            return 0
        return round(normalized_aqi * 300)
    except Exception:
        return 0
