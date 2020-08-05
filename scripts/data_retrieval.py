import os

import yaml
import sqlalchemy
import numpy as np
import pandas as pd

_DEFAULT_QUERY = \
    """
    SELECT *
    FROM {}
    WHERE "eventName" = 'Shot';
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

    # Spatial Engineered Features:
    data_df['distance_to_goal_mid'] = calc_distance_to_mid(data_df['x1'], data_df['y1'])
    data_df['distance_to_goal_nearest'] = calc_distance_to_nearest(data_df['x1'], data_df['y1'])
    data_df['angular_size_rad_goal'] = calc_angular_size_radians(data_df)
    data_df['projected_size_yds_goal'] = calc_projected_size_yds(data_df['angular_size_rad_goal'],
                                                                 data_df[
                                                                     'distance_to_goal_nearest'],
                                                                 data_df['y1'])
    data_df['kicked'] = get_kicked(data_df)
    data_df['side_of_field_matching_foot'] = compare_foot_to_side_of_field(data_df)

    # Temporal Engineered Features:
    data_df['send_off_diff'] = get_send_off_diff(data_df, sql_engine)
    data_df['free_kick_30s_ago'] = get_free_kick_data(data_df, sql_engine)

    # Cross Referenced Data
    data_df['dominant_foot'] = get_dominant_foot(data_df, sql_engine)

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
        if table.startswith('events_'):
            sql_query = query.format(table)
            response_df = query_db(engine, sql_query)
            response_df['league'] = table[7:]
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


def calc_distance_to_mid(xs: pd.Series, ys: pd.Series) -> pd.Series:
    """
    calculates the distance of an event from the center of the goal. Unfortunately soccer pitches
    are not uniformly sized so some assumptions are made currently that length and width are equal
    to the largest allowable size in international competition.  Originally I was computing in
    meters however soccer is fundamentally an imperial game ...
    :param xs: x coordinates
    :param ys: y coordinates
    :return: distances for each event.
    """
    # a confounding variable is that soccer pitches are not uniformly sized
    x_dimension = 120  # length of largest international pitch in yards
    y_dimension = 80  # width of largest international pitch in yards

    # the point (100%, 50%) being the center of the goal
    x_squared = ((xs - 100) * x_dimension / 100) ** 2  # accounting for percentage
    y_squared = ((ys - 50) * y_dimension / 100) ** 2
    distance = (x_squared + y_squared) ** (1 / 2)

    return distance


def calc_distance_to_nearest(xs: pd.Series, ys: pd.Series) -> pd.Series:
    """
    calculates the distance of an event from the nearest point on the goal line (between the goal
    posts). Unfortunately soccer pitches are not uniformly sized so some assumptions are made
    currently that length and width are equal to the largest allowable size in international
    competition.
    :param xs: x coordinates
    :param ys: y coordinates
    :return: distances for each event
    """
    # a confounding variable is that soccer pitches are not uniformly sized
    x_dimension = 120  # length of largest international pitch in yards
    y_dimension = 80  # width of largest international pitch in yards

    y_sym = ys.copy()
    # the calculation is symetric about the 50% line
    y_sym[y_sym < 50] = 100 - y_sym[y_sym < 50]

    half_goal_width_in_percent = (8 / 80) / 2

    # the point (100%, 50%) being the center of the goal
    x_squared = ((xs - 100) * x_dimension / 100) ** 2  # accounting for percentage
    y_squared = ((ys - 50) * y_dimension / 100) ** 2
    distance = (x_squared + y_squared) ** (1 / 2)

    mask_between_goal_posts = (y_sym < (50 + half_goal_width_in_percent))

    distance[mask_between_goal_posts] = 100 - xs[mask_between_goal_posts]

    return distance


def get_goal_vectors(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets vector components to each goal post
    :param shots_df: a data frame with the x1 and y1 location as columns
    :return: The components of the vectors to each goal post and to the midpoint
             x component,
             y component to 1st goal post
             y component to 2nd goal post
             y component to the middle of the goal
    """
    goal_vectors = pd.DataFrame()
    goal_width = (8 / 80) * 100  # again soccer is imperial, goals are wide 8 yds
    goal_vectors['x'] = 100 - shots_df['x1']
    goal_vectors['y1'] = 50 - shots_df['y1'] + (goal_width / 2)  # over 50%
    goal_vectors['y2'] = 50 - shots_df['y1'] - (goal_width / 2)  # under 50%
    goal_vectors['y_mid'] = 50 - shots_df['y1']

    return goal_vectors


def calc_angular_size_radians(shots_df: pd.DataFrame) -> np.ndarray:
    """
    calculates the angular size of the goal in radians
    :param shots_df: data frame of shots taker
    :return: an array of the angles
    """
    y_conversion = 80/100  # 80yds/100%
    x_conversion = 120/100 # 120yds/100%
    y_over_x_conversion = y_conversion / x_conversion
    goal_vectors = get_goal_vectors(shots_df)
    theta_1 = np.arctan(goal_vectors['y1'] / goal_vectors['x'] * y_over_x_conversion)
    theta_2 = np.arctan(goal_vectors['y2'] / goal_vectors['x'] * y_over_x_conversion)
    # x_component = goal_vectors['x']
    # y_component_1 = goal_vectors['y1']
    # y_component_2 = goal_vectors['y2']
    # theta = calc_theta(x_component, y_component_1, y_component_2)
    theta = np.abs(theta_1-theta_2)
    return theta


def calc_projected_size_yds(angular_size: pd.Series,
                            distance_to_nearest: pd.Series, ys: pd.Series) -> np.ndarray:
    """
    calculates a projected size of the goal.  This is a projection of the width of the goal in
    yards. The projection is measured along a line passing through the nearest point on the goal
    line between the posts and perpendicular to the line that connects that point to the event
    location. this ensures that the projection is never greater than 8 yards (the size of the goal)
    :param angular_size: angular size of the goal
    :param distance_to_nearest: distance to the nearest point
    :param ys: y position of the event
    :return: a projection of the goal size
    """

    # This will work for all the points outside the rectangle defined by the goal posts
    projected_size = np.abs(np.tan(angular_size)) * distance_to_nearest

    # Catching all the points between the goal posts
    mask_ys_between_goal_posts = (ys >= 45) & (ys <= 55)
    projected_size[mask_ys_between_goal_posts] = 8

    return projected_size


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
    abs_prod = (x_comp ** 2 + y1_comp ** 2) ** .5 * (x_comp ** 2 + y2_comp ** 2) ** .5
    theta = np.arccos(dot_prod / abs_prod)

    return theta


def get_kicked(shots_df: pd.DataFrame) -> pd.Series:
    """
    Returns whether the ball was kicked to score
    :param shots_df: data frame of shots
    :return: boolean whether the ball was kicked (as opposed to headed or controlled with the body)
    """
    return (shots_df['401'] | shots_df['402']) * 1  # TAG: numbers


def compare_foot_to_side_of_field(shots_df: pd.DataFrame) -> pd.Series:
    """
    Compares the foot used to the side of the field from which the shot is taken
    -1 means a mismatch, 1 means a match, and 0 means the shot was taken from the middle or the
    field (± 2% ~ half the width of the goal) or the shot was take with the head/body
    :param shots_df: data frame of shots
    :return: -1 means a mismatch, 1 means a match, and 0 means the shot was taken from the middle
             of the field (± 2% ~ half the width of the goal) or the shot was take with the
             head/body
    """
    right_matches = ((shots_df['y1'] > 52) & (shots_df['402'])) * 1  # right   # TAG: numbers
    left_matches = ((shots_df['y1'] < 48) & (shots_df['401'])) * 1  # left   # TAG: numbers
    right_does_not_match = ((shots_df['y1'] > 52) & (shots_df['401'])) * -1
    left_does_not_match = ((shots_df['y1'] < 48) & (shots_df['402'])) * -1  # left   # TAG: numbers

    aggregated = right_matches + left_matches + right_does_not_match + left_does_not_match
    return aggregated


def get_send_off_diff(shots_df: pd.DataFrame, engine: sqlalchemy.engine.Engine) -> pd.Series:
    """
    Returns the player advantage based on the red cards given
    :param shots_df: data frame of shots taken
    :param engine: sqlalchemy engine
    :return: the player differential at the time of each shot
    """
    diffs = pd.Series(data=0, index=shots_df.index)
    reds = []
    with engine.connect() as cxn:
        for league in shots_df['league'].unique():  # this is only 7 leagues
            query = """SELECT * 
                       FROM events_{} 
                       WHERE 
                        "1701" = True or 
                        "1703" = True;""".format(league)  # TAG: numbers
            reds.append(pd.read_sql(query, cxn))

    reds_df = pd.concat(reds)
    for match in reds_df['matchId'].unique():  # 305 matches with red cards
        for ind, row in shots_df[shots_df['matchId'] == match].iterrows():  # 50 max
            same_team_slice = reds_df[(reds_df['eventSec'] < row['eventSec']) &
                                      (reds_df['teamId'] == row['teamId'])]
            other_team_slice = reds_df[(reds_df['eventSec'] < row['eventSec']) &
                                       (reds_df['teamId'] != row['teamId'])]
            reds_same = len(same_team_slice[same_team_slice['matchId'] == match])
            reds_other = len(other_team_slice[other_team_slice['matchId'] == match])
            diffs[ind] = reds_other - reds_same

    return diffs


def get_dominant_foot(shots_df: pd.DataFrame, engine: sqlalchemy.engine.Engine) -> pd.Series:
    """
    Cross-references the player database to see if the shot was taken with the player's dominant
    foot. The values mapped are as follows:
        -1 for non-dominant foot kick,
        0  for head/body,
        +1 for  dominant foot kick

    :param shots_df: data frame of shots taken
    :param engine: sqlalchemy engine
    :return: series for the shots taken of:

    """

    query = """SELECT "wyId", "foot" 
               FROM players;"""

    with engine.connect() as cxn:
        player_foot = pd.read_sql(query, cxn)

    player_foot_dict = pd.Series(player_foot['foot'], index=player_foot['wyId']).to_dict()

    def foot_mapping_func(row) -> int:
        """
        mapping function because it was too confusing to use lambda
        only one (left, right, or head) should trigger so to get the correct range we need weight
        them appropriately, if none trigger, then not a dominant foot, so it should return -1
        if head triggers, then it returns to 0
        if left or right trigger then it returns +1
        :param row: needs columns: '401', '402', '403', and 'playerId' to be in the row
        :return: integer representation of whether it is a dominant foot
        """
        left = row['401'] & (player_foot_dict.get(row['playerId'], 0) in ['left', 'both'])
        right = row['402'] & (player_foot_dict.get(row['playerId'], 0) in ['right', 'both'])
        head = row['403']  # TAG: numbers

        return (2 * left) + (2 * right) + (1 * head) - 1

    return shots_df.apply(foot_mapping_func, axis=1)


def get_free_kick_data(shots_df, engine) -> pd.Series:
    """
    gets free kick data from the last 30 sec
    :param shots_df: data frame of shots taken
    :param engine: sqlalchemy engine
    :return: a series of "subEventName" of free kicks that happen within 30 seconds of a shot
    """
    query = """ SELECT
                    response."matchId", 
                    response."eventSec",
                    response."time_to_last_event",
                    response."previous_event",
                    response."previous_team"
                FROM (
                    SELECT "eventName", "matchId", "eventSec",
                        LAG("subEventName", 1) OVER 
                            (PARTITION BY "matchId" ORDER BY "eventSec") AS "previous_event",
                        "teamId",
                        LAG("teamId", 1) OVER 
                            (PARTITION BY "matchId" ORDER BY "eventSec") AS "previous_team",
                        "eventSec" -LAG("eventSec", 1) OVER 
                            (PARTITION BY "matchId" ORDER BY "eventSec") AS "time_to_last_event"
                    FROM    events_{}
                    WHERE   "eventName" = 'Shot' OR 
                            "eventName" = 'Free Kick'
                    ) AS response
                WHERE 
                      "eventName" = 'Shot';"""
    shots_prev_event_list = []
    with engine.connect() as cxn:
        for league in shots_df['league'].unique():
            shots_prev_event_list.append(pd.read_sql(query.format(league), cxn))

    shots_prev_event_df = pd.concat(shots_prev_event_list)

    # Merging to make sure the indices align
    merge_on = ['matchId', 'eventSec']
    shots_merged = pd.merge(shots_df, shots_prev_event_df, how='left', on=merge_on)

    # Mask:
    #   - less than 30 sec ago
    #   - previous event is not a shot aka its a free kick
    #   - make sure its the same team
    mask = (shots_merged['time_to_last_event'] < 30) & \
           (shots_merged['previous_event'] != 'Shot') & \
           (shots_merged['previous_team'] == shots_merged['teamId'])

    return mask * shots_merged['previous_event']


if __name__ == '__main__':
    df = get_data()
    print(df[['projected_size_yds_goal']].describe())
    print(df.head())
    print(df.columns)
    print(df['projected_size_yds_goal'].max())

