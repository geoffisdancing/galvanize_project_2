import pandas as pd
import numpy as np


def load_intraday():
    user_id = []
    date = []
    minute = []
    steps = []
    resource = []

    with open('data/meas_fitbit_intraday.txt', 'r') as f:
        next(f)
        for line in f:
            splits = line.split('|')
            user_id.append(int(splits[1]))
            date.append(splits[3])
            minute.append(int(splits[4]))
            steps.append(int(splits[5]))
            resource.append(splits[6])
    df = pd.DataFrame({'user_id':user_id, 'date':date, 'minute':minute, 'steps':steps, 'resource':resource})
    df2 = df.ix[:,[4,0,1,3,2]]
    df2['date']=pd.to_datetime(df2['date'], format='%d%b%Y') #convert fitbit date to datetime
    return df2



'''
id|user_id|measurement_id|date|minute|value|resource|device
18|17|3591573|02APR2013|312|18|steps|Flex
19|17|3591573|02APR2013|314|26|steps|Flex
20|17|3591573|02APR2013|316|51|steps|Flex
21|17|3591573|02APR2013|317|22|steps|Flex
22|17|3591573|02APR2013|319|54|steps|Flex
23|17|3591573|02APR2013|320|7|steps|Flex
24|17|3591573|02APR2013|329|21|steps|Flex
25|17|3591573|02APR2013|330|25|steps|Flex
26|17|3591573|02APR2013|366|33|steps|Flex
27|17|3591573|02APR2013|368|16|steps|Flex
28|17|3591573|02APR2013|369|16|steps|Flex
29|17|3591573|02APR2013|370|6|steps|Flex

'''
