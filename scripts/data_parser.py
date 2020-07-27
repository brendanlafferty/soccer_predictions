import yaml
import json
import pandas as pd
import sqlalchemy
from warnings import warn

with open('../keys/sql_cred.yml') as fp:
    sql_dict = yaml.load(fp)

# Establish connection with soccer_data and get tables
engine = sqlalchemy.create_engine(sql_dict['sql_url'])
tables = engine.table_names()

leagues = ['England', 'France', 'Germany', 'Italy', 'Spain', 'European_Championship', 'World_Cup']
file_path_generic = '../data/events/events_{}.json'
leagues_to_load = list(leagues)
table_template = 'events_{}'

# Check to see if the data has been loaded
for league in leagues:
    if table_template.format(league.lower()) in tables:
        print(f'{league} is already loaded it will be skipped')
        leagues_to_load.remove(league)
print('The following leagues will be loaded:')
for league in leagues_to_load:
    print(league)

# Load league data:
for league in leagues_to_load:
    league_data_fp = file_path_generic.format(league)
    print(f'opening file for: {league}')
    with open(league_data_fp) as f_stream:
        events = json.load(f_stream)
    print('json loaded')
    print(f'{len(events)} events found\n')

    print('Starting flattening process:')
    for event in events:
        print(f'Working on event: {event["id"]}')
        print(f'\t{event["eventName"]}')
        # parse positional data
        y1 = event['positions'][0]['y']
        x1 = event['positions'][0]['x']
        try:
            y2 = event['positions'][1]['y']
        except IndexError:
            x2 = y2 = None
            warn('No second position! setting values to None')
        else:
            x2 = event['positions'][1]['x']
        event['y1'] = y1
        event['x1'] = x1
        event['y2'] = y2
        event['x2'] = x2

        # parse tags
        for item in event['tags']:  # max length ~ 6 items
            event.update({value: True for value in [*item.values()]})

        # remove data so the the columns aren't there for the construction of the dataframe
        del event['positions']
        del event['tags']

    print('\n\nFlattening Complete\n\nCreating DF:')
    events_df = pd.json_normalize(events)
    print('DF complete:')
    # print(events_df.head(10))

    table_name = table_template.format(league.lower())
    print(f'writing to sql db: {table_name}')
    with engine.connect() as cxn:
        events_df.to_sql(table_name, cxn, index=False)

    print(f'table "{table_name}" stored\n\n')
print('Completed loading the following league data:')
for league in leagues_to_load:
    print(league)
