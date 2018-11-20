import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from dataset import *

def parseData(train_data): 

	train_data['name'].fillna(" ")
	train_data['desc'].fillna(" ")

	date_column = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
	for i in date_column:
		train_data[i]=train_data[i].apply(lambda x: datetime.fromtimestamp(int(x)).strftime("%Y-%m-%d %H:%M:%S"))

	train_data['log_goal'] = np.log(train_data['goal'])

	train_data['launched_month'] = train_data['launched_at'].apply(lambda dt: dt[5:7])
	train_data['launched_year'] = train_data['launched_at'].apply(lambda dt: dt[0:4])
	train_data['launched_quarter'] = train_data['launched_at'].apply(lambda dt: countQuarter(dt))

	train_data['duration'] = train_data[['launched_at', 'deadline']].apply(lambda dt: measureDuration(dt), axis=1)
	train_data['duration_weeks'] = train_data['duration'].apply(lambda dt: measureDurationByWeek(dt))

	return train_data





