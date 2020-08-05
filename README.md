# soccer_predictions
A classifier that predicts whether a shot will result in a goal


## Goal of the classifier
The goal is to determine what kinds of features are predictive of goal scoring to inform strategic 
decision making.


## Main Files

- [models.ipynb](models.ipynb)\
  notebook containing all the modeling
  
- [load_events_db.py](scripts/load_events_db.py) [load_player_db.py](scripts/load_player_db.py) \
  Scripts for loading already downloaded json files into a postgres database
  
- [data_retrieval.py](scripts/data_retrieval.py) \
  functions for loading the data from the database and engineer the appropriate features
  
- [paper_functions.py](scripts/paper_functions.py) \
  Selected functions from the paper see 
  [figshare link](https://figshare.com/articles/software/Plots_replication_code_of_Nature_Scientific_Data_paper/11473365)

## Caution
There are many ways to go astray in soccer analysis. Maybe a bit over used is the example of Charles 
Reep ([wikipedia](https://en.wikipedia.org/wiki/Charles_Reep) 
[fivethirtyeight](https://fivethirtyeight.com/features/how-one-mans-bad-math-helped-ruin-decades-of-english-soccer/))

 
With that in mind there are a few things to keep in mind not to do:

- The goal is not to overload player so that they freeze in the moment in informational overload.
 Players shouldn't take this information and apply blindly. Bad application: 
  - Shooting with the foot leads to higher likelihood of a goal. Players should always prefer 
    shooting with feet (over body or head). Players should use bicycle kicks over heading the ball. 

  This is a bad take away.
  
- The model is not built to assess whether a particular player bombed a chance. and shouldn't be 
  used to assess player performance. 
  - The model says that Wondo should have scored against Belgium. Wondo let us down
  
  Again not the goal
  

## Big picture takeaways
- Positional information is the biggest predictor of goal likelihood. Getting closer to goal is 
  better. Well if I'm honest this is a no-brainer point. 
- Kicking the ball had a higher likelihood of scoring a goal (over using the head or body).  This 
  could be from a number of factors, it is easier to control the ball with the feet, balls 
  controlled with the feet are more likely to be shot from the ground which are easier to control, 
  etc. This is brought up to signal that the takeaway for players should not be "only use my feet". 
  Instead this should be thought of in a strategic sense. The goal should be to create chances that 
  can be shot with the feet.  This mainly applies to shots in play and does not necessarily extend 
  to set plays. 
- The player using their dominant foot was not a significant factor in predicting the likelihood of 
  a goal.  I think this is the one takeaway that players can use "in the moment" so to speak.  Using
  the non-dominant foot is not a significant detriment.  (I'll add a big caveat to this one the 
  model uses the top professional athletes as its base case, they are much better than the average 
  player however I do think there is something to this takeaway)
  
  
## Aspirations for the future

My hope is that I can add more features to explore in this data set.  On a relatively quick turn 
around I engineered 9 features; however this is just scratching the surface as to what this dataset 
has to offer. Some off the cuff thoughts on interesting features to explore:
 - number of passes leading up to the shot
 - what direction the ball comes from (pass/cross) relative to the where the shot is taken
 - whether the shot resulted from a turnover in the shot takers attacking half
 - whether the shot was assisted
 
Going a bit further another level to analysing this data set would be to assess free kicks and 
whether it makes sense to shoot or cross the ball. 


### Data source:
 - Pappalardo, L., Cintia, P., Rossi, A. et al. A public data set of spatio-temporal match events in soccer competitions. Sci Data 6, 236 (2019). https://doi.org/10.1038/s41597-019-0247-7
   -  [Figshare Collection](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5)



###### Repo:
[brendan_lafferty/soccer_predictions](https://github.com/brendanlafferty/soccer_predictions)