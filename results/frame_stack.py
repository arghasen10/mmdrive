import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


plt.xlabel('Number of Frames')
plt.ylabel('Weighted F1-Score')
plt.show()