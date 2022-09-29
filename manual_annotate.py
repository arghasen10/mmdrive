import csv
from datetime import datetime, timedelta

starttime = datetime.strptime('2022_09_25_19:54:02', '%Y_%m_%d_%H:%M:%S')
endtime = datetime.strptime('2022_09_25_20:07:57', '%Y_%m_%d_%H:%M:%S')

with open('driving_activityAnnot_2022_09_25_19_54_02.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['datetime', 'activity'])
    delta = timedelta(seconds=1)

    while starttime <= endtime:
        writer.writerow([starttime, 'Normal driving'])
        starttime += delta

