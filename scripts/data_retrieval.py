import yaml
import sqlalchemy
import pandas as pd

_DEFAULT_QUERY = \
    """
    SELECT *
    FROM {}
    WHERE "eventName" = 'Shot'
    """


def get_data():
    """

    :return:
    """
    engine = get_engine()
    data_df = assemble_df(engine)
    data_df['distance_to_goal'] = calc_distances(data_df)
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
    with open('../keys/sql_cred.yml') as fp:
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
    to the largest allowable size in international competition
    :param events_df: dataframe of events
    :return: distances for each event.
    """
    # a confounding variable is that soccer pitches are not uniformly sized
    x_dimension = 110  # length of largest international pitch in meters
    y_dimension = 75  # width of largest international pitch in meters

    x_squared = ((events_df['x1'] - 100) * x_dimension / 100) ** 2  # accounting for percentage
    y_squared = ((events_df['y1'] - 50) * y_dimension / 100) ** 2
    distances = (x_squared + y_squared)**(1/2)

    return distances


if __name__ == '__main__':
    df = get_data()
    print(df.head())
