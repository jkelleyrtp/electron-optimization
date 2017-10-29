import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from math import exp

_c0 = (1.0-1.0/(np.linspace(0.0001,15,10000)+1.0))

plt.figure()
plt.plot(_c0)
plt.show()
