import os

import yaml
import sqlalchemy
import numpy as np
import pandas as pd

_DEFAULT_QUERY = \
    """
    SELECT *
    FROM {}
    WHERE "eventName" = 'Shot'
    """


def get_data() -> pd.DataFrame:
    """
    main function that gets the raw data all engineered features
    :return: data frame of data and engineered features
    """
    sql_engine = get_engine()
    data_df = assemble_df(sql_engine)
    # Clean data
    data_df.replace([None], False, inplace=True)

    # Engineered Features:
    distances = calc_distances(data_df)
    data_df['distance_to_goal'] = distances
    goal_vectors = get_goal_vectors(data_df)
    data_df['angular_size_rad_goal'] = calc_angular_size_radians(goal_vectors)
    data_df['projected_size_yds_goal'] = cal_projected_size_yds(goal_vectors, distances)

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
    with open(sql_cred_loc) as file_stream:
        sql_dict = yaml.safe_load(file_stream)

    return sql_dict['sql_url']


def get_sql_cred_location() -> str:
    """
    Gets the absolute path to the helper file
    :return: absolute path to helper file
    """
    filename = os.path.abspath(__file__)
    root_folder = filename[:filename.find('/scripts/')]

    return os.path.join(root_folder, 'keys/sql_cred.yml')


def assemble_df(engine: sqlalchemy.engine.Engine, query: str = None) -> pd.DataFrame:
    """
    queries every table in the data base with the same query
    :param engine: sqlalchemy engine for the connection
    :param query: sql query sting. if None the default one will be used
    :return: the results of the queries concatenated into 1 data frame
    """
    if not query:
        query = _DEFAULT_QUERY

    tables = engine.table_names()
    data_dfs = []

    for table in tables:
        sql_query = query.format(table)
        response_df = query_db(engine, sql_query)
        data_dfs.append(response_df)

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
    :param events_df: data frame of events
    :return: distances for each event.
    """
    # a confounding variable is that soccer pitches are not uniformly sized
    x_dimension = 120  # length of largest international pitch in yards
    y_dimension = 80  # width of largest international pitch in yards

    x_squared = ((events_df['x1'] - 100) * x_dimension / 100) ** 2  # accounting for percentage
    y_squared = ((events_df['y1'] - 50) * y_dimension / 100) ** 2
    distances = (x_squared + y_squared)**(1/2)

    return distances


def get_goal_vectors(events: pd.DataFrame) -> pd.DataFrame:
    """
    Gets vector components to each goal post
    :param events: a data frame with the x1 and y1 location as columns
    :return: The components of the vectors to each goal post and to the midpoint
             x component,
             y component to 1st goal post
             y component to 2nd goal post
             y component to the middle of the goal
    """
    goal_vectors = pd.DataFrame()
    goal_width = (8/80)*100  # again soccer is imperial, goals are wide 8 yds
    goal_vectors['x'] = 100 - events['x1']
    goal_vectors['y1'] = 50 - events['y1'] + (goal_width / 2)
    goal_vectors['y2'] = 50 - events['y1'] - (goal_width / 2)
    goal_vectors['y_mid'] = 50 - events['y1']

    return goal_vectors


def calc_angular_size_radians(goal_vectors: pd.DataFrame) -> np.ndarray:
    """
    calculates the angular size of the goal in radians
    :param goal_vectors: goal_vector data frame output from get_goal_vectors
    :return: an array of the angles
    """
    x_component = goal_vectors['x']
    y_component_1 = goal_vectors['y1']
    y_component_2 = goal_vectors['y2']
    theta = calc_theta(x_component, y_component_1, y_component_2)

    return theta


def cal_projected_size_yds(goal_vectors: pd.DataFrame, distances: pd.Series):
    """
    calculates a projected size of the goal.
    :param goal_vectors:
    :param distances:
    :return:
    """
    x_component = goal_vectors['x']
    y_component_1 = goal_vectors['y1']
    y_component_2 = goal_vectors['y2']
    y_component_mid = goal_vectors['y_mid']
    theta_1 = calc_theta(x_component, y_component_mid, y_component_1)
    theta_2 = calc_theta(x_component, y_component_2, y_component_mid)
    projections_1 = np.abs(np.tan(theta_1)) * distances
    projections_2 = np.abs(np.tan(theta_2)) * distances
    projected_sizes = projections_1 + projections_2
    return projected_sizes


def calc_theta(x_comp, y1_comp, y2_comp) -> np.ndarray:
    """
    using the components of 2 vectors this function calculates the angle between the 2 vectors.
    assumes a shared x component
    :param x_comp: a shared x component to the goal
    :param y1_comp: the first y component
    :param y2_comp: the second y component
    :return: the angular width of the goal
    """
    dot_prod = x_comp * (y1_comp + y2_comp)
    abs_prod = (x_comp**2 + y1_comp**2)**.5 * (x_comp**2 + y2_comp**2)**.5
    theta = np.arccos(dot_prod/abs_prod)
    return theta


if __name__ == '__main__':
    df = get_data()
    print(df.describe())
    print(df.head())
