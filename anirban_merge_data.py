import csv
from datetime import datetime, timedelta

filename = '20220920_23_12_33.log'
starttime = datetime.strptime(filename.split('.')[0], "%Y%m%d_%H_%M_%S")
day = '2022-09-20'
delta = timedelta(seconds=1)

with open('activityAnnot_2022_09_20_23_12_33.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['datetime', 'activity'])

with open(filename, 'r') as file:
    lines = file.readlines()
    for l in lines:
        s_t, e_t, act = l.strip().split(',')
        s_t = day + ' ' + s_t
        s_t = datetime.strptime(s_t, '%Y-%m-%d %H:%M:%S')
        e_t = day + ' ' + e_t
        e_t = datetime.strptime(e_t, '%Y-%m-%d %H:%M:%S')
        with open('activityAnnot_2022_09_20_23_12_33.csv', 'a') as file2:
            writer = csv.writer(file2)
            while starttime <= s_t:
                writer.writerow([starttime, 'Normal driving'])
                starttime += delta
            while starttime <= e_t:
                writer.writerow([starttime, act])
                starttime += delta
