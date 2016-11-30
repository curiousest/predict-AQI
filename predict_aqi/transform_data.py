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

