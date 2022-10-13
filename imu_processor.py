from datetime import datetime, timedelta
import pandas as pd
import glob
import time
import csv
from dateutil.relativedelta import relativedelta
import json
import numpy as np
from vmdpy import VMD
import matplotlib.pyplot as plt
import statsmodels.api as sm
T = 1000
fs = 1/T
t = np.arange(1,T+1)/T
freqs = 2 * np.pi * (t-0.5-fs) / fs


files = glob.glob('~/Desktop/dangerous_driving_a/data25_09_22/s*_25_09_22/nexar_data/*B.dat')
files.sort()
plot_imu = []
time_then = []
for f in files:
    with open(f, 'r') as file:
        print(f)
        lines = file.readlines()
        for l in lines:
            data = l.strip().split(',')
            if len(data) == 9:
                time_val = int(data[0].split('|')[0]) / 1000000
                actual_time = datetime.strptime(time.strftime('%Y%m%d%H%M%S', time.localtime(time_val)), '%Y%m%d%H%M%S') + \
                              timedelta(days=14, seconds=-3)
                if datetime.strptime("2022-09-25 14:21:00", "%Y-%m-%d %H:%M:%S") < actual_time < \
                        datetime.strptime("2022-09-25 14:34:00", "%Y-%m-%d %H:%M:%S"):
                    time_then.append(actual_time)
                    plot_imu.append(float(data[4]))
plot_imu = np.array(plot_imu)


plt.plot(time_then, plot_imu)
# plt.xticks()
plt.show()

alpha = 2000       # moderate bandwidth constraint
tau = 0            # noise-tolerance (no strict fidelity enforcement)
K = 2              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7

u, u_hat, omega = VMD(plot_imu, alpha, tau, K, DC, init, tol)

# files = glob.glob('~/Desktop/dangerous_driving_a/data25_09_22/s*_25_09_22/mmwave_data/*.txt')
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
#
#
# plot_noise = np.array(plot_noise)
# print(plot_noise.shape, plot_imu.shape)
#. Visualize decomposed modes
new_df = pd.DataFrame(['Datetime', 'IMU', 'mmWave'])
df1 = pd.read_csv('imu_data.csv')
df1['Datetime'] = pd.to_datetime(df1['Datetime'])
imu_data = df1.groupby(df1.Datetime.dt.time).mean()
# u, u_hat, omega = VMD(imu_data, alpha, tau, K, DC, init, tol)
new_df['IMU'] = (u[1] > 4)*10
new_df['Datetime'] = df1.Datetime.dt.time
df = pd.read_csv('noise_data.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
noise_data = df.groupby(df.Datetime.dt.time).mean()

# correlation = sm.tsa.stattools.ccf(noise_data, (u[1] > 4)*10, adjusted=False)
# plt.plot(correlation)
# plt.show()
print(u.shape, noise_data.shape)
