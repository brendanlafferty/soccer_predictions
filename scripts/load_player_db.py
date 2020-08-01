import json

import yaml
import sqlalchemy
import pandas as pd

with open('../keys/sql_cred.yml') as file_stream:
    sql_dict = yaml.safe_load(file_stream)

engine = sqlalchemy.create_engine(sql_dict['sql_url'])
if 'players' in engine.table_names():
    print('Data already loaded')
else:

    with open('../data/players.json') as f_stream:
        players = json.load(f_stream)

    players_df = pd.json_normalize(players)

    with engine.connect() as cxn:
        players_df.to_sql('players', cxn, index=False)

