import csv
from datetime import datetime, timedelta

starttime = datetime.strptime('2022_10_10_23:28:06', '%Y_%m_%d_%H:%M:%S')
endtime = datetime.strptime('2022_10_10_23:29:38', '%Y_%m_%d_%H:%M:%S')

with open('driving_activityAnnot_2022_10_10_23_28_06.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['datetime', 'activity'])
    delta = timedelta(seconds=1)

    while starttime <= endtime:
        writer.writerow([starttime, 'Normal driving'])
        starttime += delta

