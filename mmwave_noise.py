import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
# from datetime import datetime, timedelta
# from dateutil.relativedelta import relativedelta
# import glob
# import time
# import csv
# import matplotlib.pyplot as plt
# import numpy as np
# import json
# files = glob.glob('~/Desktop/dangerous_driving_a/data25_09_22/sugandh_25_09_22/mmwave_data/*.txt')
# files.sort()
# plot_noise = []
# time_then = []
# for f in files:
#     with open(f, 'r') as file:
#         print(f)
#         lines = file.readlines()
#         for l in lines:
#             try:
#                 data = json.loads(l)
#                 datenow = data['answer']['datenow']
#                 timenow = data['answer']['timenow']
#                 # datetime.strptime(time.strftime('%Y%m%d%H%M%S', time.localtime(time_val)), '%Y%m%d%H%M%S') + \
#                 # timedelta(days=14, seconds=-3)
#                 date_obj = datetime.strptime(datenow+ ' ' + timenow, "%d/%m/%Y %H_%M_%S")
#                 result_1 = date_obj + relativedelta(months=+1)
#                 if datetime.strptime("2022-09-25 14:21:00", "%Y-%m-%d %H:%M:%S") < result_1 < \
#                         datetime.strptime("2022-09-25 14:34:00", "%Y-%m-%d %H:%M:%S"):
#                     print(result_1)
#                     time_then.append(result_1)
#                     noise_profile = np.array(data['answer']['noiserp_y'])
#                     plot_noise.append(noise_profile.mean())
#             except Exception as e:
#                 continue
    # break
# plt.plot(time_then, plot_noise)
# plt.xticks()
# plt.show()
#
# with open('noise_data.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Datetime', 'IMU'])
#
# for d,i in zip(time_then[1:], plot_noise):
#     with open('noise_data.csv', 'a') as file:
#         writer = csv.writer(file)
#         writer.writerow([d, i])

df = pd.read_csv('noise_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
# noise_data = \
df.groupby(df.Datetime.dt.time).mean().plot()
df1 = pd.read_csv('imu_data.csv')
df1['Datetime'] = pd.to_datetime(df1['Datetime'])
# imu_data =
df1.groupby(df1.Datetime.dt.time).mean().plot()
# imu_data = imu_data[:noise_data.shape[0],]
# correlation = sm.tsa.stattools.ccf(noise_data, imu_data, adjusted=False)
# plt.plot(correlation)
plt.show()
