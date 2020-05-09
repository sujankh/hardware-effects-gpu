import os
import sys

import matplotlib.pyplot as plt
import seaborn

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT)
from utils import benchmark


keys = ["Offset"]
values = [0, 1, 2, 4, 8, 16, 32, 36]

frame = benchmark(keys, values)

g = seaborn.barplot(data=frame, x="Offset", y="Time")
plt.show()
