import math
import itertools
import numpy as np
from operator import itemgetter

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


def generate_time_inputs(df):
    df['minute_of_day_sin'], df['minute_of_day_cos'], \
    df['day_of_year_sin'], df['day_of_year_cos'], \
    df['day_of_week_sin'], df['day_of_week_cos'], \
    df['day_of_month_sin'], df['day_of_month_cos'] = \
        zip(*df["measurement_datetime"].map(get_normalized_time_inputs))
    time_columns = ['minute_of_day_sin', 'minute_of_day_cos', 'day_of_year_cos', 'day_of_year_sin', 'day_of_week_cos',
               'day_of_week_sin', 'day_of_month_sin', 'day_of_month_cos']
    return df, time_columns


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


def shift_and_save_column(df, source_col_name, dest_col_name, shift=1):
    df[dest_col_name] = getattr(df, source_col_name).shift(shift)


def shift_inputs_backwards(x_all, shift_count, source_column_name, column_string_to_format='{}_ago'):
    '''
    Returns x_all dataframe with `shift_count` new columns shifting the source column backwards one row for each new
    column and returns the column names.
    '''
    feature_columns = []
    shift_and_save_column(x_all, source_column_name, column_string_to_format.format('0'))
    feature_columns.append(column_string_to_format.format('0'))

    for i in range(1, shift_count):
        prev_feature_column = column_string_to_format.format(str(i))
        feature_column = column_string_to_format.format(str(i + 1))
        feature_columns.append(feature_column)
        shift_and_save_column(x_all, prev_feature_column, feature_column)
    return x_all, feature_columns


def shift_outputs_forwards(y_all, shift_count, source_column_name, column_string_to_format='{}_ahead'):
    shift_and_save_column(y_all, source_column_name, column_string_to_format.format('1'), shift=-1)
    output_columns = [column_string_to_format.format('1')]
    for i in range(2, shift_count + 1):
        prev_input_column = column_string_to_format.format(str(i - 1))
        output_column = column_string_to_format.format(str(i))
        output_columns.append(output_column)
        shift_and_save_column(y_all, prev_input_column, output_column, shift=-1)
    return y_all, output_columns


def clean_data(df, input_columns):
    df['all_input_equal'] = df[input_columns].apply(lambda x: len(set(x)) < 3, axis=1)
    return df[df['all_input_equal'] == False]


def row_has_same_time(row, number_of_cities):
    '''
    Returns whether the row has all times within 10 minutes of the base city
    '''
    ten_minutes = 60 * 10 * 10**9
    for city_number in range(2, number_of_cities + 1):
        diff = abs(
            row['city_1_measurement_datetime'].value -
            row['city_{}_measurement_datetime'.format(city_number)].value
        )
        if diff > ten_minutes:
            return False
    return True


def get_city_with_smallest_time(row, number_of_cities):
    row_names = ['city_{}_measurement_datetime'.format(i)
                 for i in range(1, number_of_cities + 1)]
    return min(enumerate(row[row_names]), key=itemgetter(1))[0] + 1


def city_has_no_more_data(row, city_index):
    return np.isnan(row['city_{}_id'.format(city_index)])


def shift_single_loc_up(df, shift_index, loc_index):
    loc_columns = [s.format(loc_index) for s in
                    ['loc_{}_id', 'loc_{}_measurement_datetime', 'loc_{}_aqi']]
    other_columns = [col for col in df.columns if col not in city_columns]
    shifted_df = df[other_columns][shift_index:].join(
        df[loc_columns][shift_index:].shift(-1)
    )
    return df[:shift_index].append(shifted_df)


def align_multi_location_time_series_data(df, number_of_locations):
    continuous_time_series = []
    start_index = 0
    current_index = 0
    while df.count()[0] > current_index:
        current_index += 1
        row = df.loc[current_index]
        if not row_has_same_time(row, number_of_locations):
            continuous_time_series.append((start_index, current_index))
            while not row_has_same_time(row, number_of_locations):
                loc_to_shift = get_city_with_smallest_time(row, number_of_locations)
                shift_single_loc_up(df, current_index, loc_to_shift)
                row = df.loc[current_index]
                if city_has_no_more_data(row, loc_to_shift):
                    return df, continuous_time_series
            start_index = current_index
    continuous_time_series.append((start_index, current_index))
    return df, continuous_time_series
