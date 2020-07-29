import os
import yaml
import sqlalchemy
import numpy as np
import pandas as pd
from typing import Tuple

_DEFAULT_QUERY = \
    """
    SELECT *
    FROM {}
    WHERE "eventName" = 'Shot'
    """
# location vector components
LocationVectorComps = Tuple[pd.Series, pd.Series, pd.Series]


def get_data() -> pd.DataFrame:
    """
    main function that gets the raw data all engineered features
    :return: dataframe of data and engineered features
    """
    sql_engine = get_engine()
    data_df = assemble_df(sql_engine)
    # Clean data
    data_df.replace([None], False, inplace=True)

    # Engineered Features:
    data_df['distance_to_goal'] = calc_distances(data_df)
    data_df['apparent_size_rad'] = calc_apparent_size_radians(data_df)

    return data_df


def get_engine() -> sqlalchemy.engine.Engine:
    """
    creates a sqlalchemy engine
    :return: a sqlalchemy engine
    """
    db_string = get_db_location()
    engine = sqlalchemy.create_engine(db_string)

    return engine


def get_db_location() -> str:
    """
    Opens the sql_cred.yml helper file to get the location of the db and returns the location as
    a string ready for sqlalchemy to connect to
    :return: database location string
    """
    sql_cred_loc = get_sql_cred_location()
    with open(sql_cred_loc) as fp:
        sql_dict = yaml.safe_load(fp)

    return sql_dict['sql_url']


def assemble_df(engine: sqlalchemy.engine.Engine, query: str = None) -> pd.DataFrame:
    """
    queries every table in the data base with the same query
    :param engine: sqlalchemy engine for the connection
    :param query: sql query sting. if None the default one will be used
    :return: the results of the queries concatenated into 1 dataframe
    """
    if not query:
        query = _DEFAULT_QUERY

    tables = engine.table_names()
    data_dfs = []

    for table in tables:
        sql_query = query.format(table)
        df = query_db(engine, sql_query)
        data_dfs.append(df)

    data_df = pd.concat(data_dfs)
    return data_df


def query_db(engine: sqlalchemy.engine.Engine, query: str = None) -> pd.DataFrame:
    """
    Queries a data base with the string and returns the resulting dataframe
    :param engine: sqlalchemy engine used to make a connection
    :param query: sql query string, if not provide a default query will be used
    :return: resulting data frame from the query
    """
    if not query:
        query = _DEFAULT_QUERY.format('England')

    with engine.connect() as cxn:
        data = pd.read_sql(query, cxn)

    return data


def calc_distances(events_df: pd.DataFrame) -> pd.Series:
    """
    calculates the distance of an event from the center of the goal. Unfortunately soccer pitches
    are not uniformly sized so some assumptions are made currently that length and width are equal
    to the largest allowable size in international competition.  Originally I was computing in
    meters however soccer is fundamentally an imperial game ...
    :param events_df: dataframe of events
    :return: distances for each event.
    """
    # a confounding variable is that soccer pitches are not uniformly sized
    x_dimension = 120  # length of largest international pitch in yards
    y_dimension = 80  # width of largest international pitch in yards

    x_squared = ((events_df['x1'] - 100) * x_dimension / 100) ** 2  # accounting for percentage
    y_squared = ((events_df['y1'] - 50) * y_dimension / 100) ** 2
    distances = (x_squared + y_squared)**(1/2)

    return distances


def calc_apparent_size_radians(events_df: pd.DataFrame) -> np.ndarray:
    """
    calculates the apparent size of the goal in radians
    :param events_df: events df to calculate goal size
    :return: an array of the
    """
    goal_vectors = get_goal_vectors(events_df[['x1', 'y1']])
    theta = calc_theta(goal_vectors)

    return theta


def get_sql_cred_location() -> str:
    """
    Gets the absolute path to the helper file
    :return: absolute path to helper file
    """
    filename = os.path.abspath(__file__)
    root_folder = filename[:filename.find('/scripts/')]

    return os.path.join(root_folder, 'keys/sql_cred.yml')


def get_goal_vectors(location_vector: pd.DataFrame) -> LocationVectorComps:
    """
    Gets vector components to each goal post
    :param location_vector: a data frame with the x1 and y1 location as columns
    :return: The components of the vectors to each goal post
             x component,
             y component to 1st goal post
             y component to 2nd goal post
    """
    goal_width = (8/80)*100  # again soccer is imperial, goals are wide 8 yds
    goal_vector_x = 100 - location_vector['x1']
    goal_vector_y1 = 50 - location_vector['y1'] + (goal_width/2)
    goal_vector_y2 = 50 - location_vector['y1'] - (goal_width/2)

    return goal_vector_x, goal_vector_y1, goal_vector_y2


def calc_theta(goal_vectors: LocationVectorComps) -> np.ndarray:
    """
    using the goal vectors from get_goal_vectors this function calculates the angle between the 2
    vectors aka the apparent size of the goal in radians.
    :param goal_vectors:
    :return:
    """
    dot_prod = goal_vectors[0] * (goal_vectors[1] + goal_vectors[2])
    abs_prod = (goal_vectors[0].pow(2) + goal_vectors[1].pow(2))**.5 * \
               (goal_vectors[0].pow(2) + goal_vectors[2].pow(2))**.5
    theta = np.arccos(dot_prod/abs_prod)
    return theta


if __name__ == '__main__':
    df = get_data()
    print(df.describe())
    print(df.head())
