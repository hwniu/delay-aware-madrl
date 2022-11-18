import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

delay_data = pd.read_excel("./oai_时延测试数据.xlsx")

plt.plot(delay_data["序号"], delay_data["时延/ms"], linewidth=0.5, color="blue")
plt.xlabel("Time /10s")
plt.ylabel("Transmission Delay /ms")
# plt.savefig("oai_delay_figure1.jpg", dpi=1000, bbox_inches = 'tight')
plt.show()

bins = [i*5 for i in range(20)]
ratio = []
for index in range(len(bins)):
    ratio_number = 0
    for item in delay_data["时延/ms"]:
        if index >= 1:
            if bins[index - 1] < item < bins[index]:
                ratio_number = ratio_number + 1
    ratio.append(ratio_number)
for i in range(len(ratio)):
    plt.bar(i, ratio[i])
my_x_ticks = np.arange(0, len(ratio), 1)
plt.xticks(my_x_ticks)
plt.xlabel("Transmission Delay Interval / 5ms")
plt.ylabel("Frequency")
# plt.savefig("oai_delay_figure2.jpg", dpi=1000, bbox_inches = 'tight')
plt.show()
