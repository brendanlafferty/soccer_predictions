# Keys



###### sql_cred.yml
This is where the sql connection info get stored. it should look something like this: 
```yaml
sql_url: postgresql://user:pw@localhost:5432/soccer_data
```
This file is both by [data_parser.py](../scripts/load_events_db.py) for loading the database from the 
json data and [data_retrieval.py](../scripts/data_retrieval.py) for accessing the data