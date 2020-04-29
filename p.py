import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd

d1 = {'c1': [np.random.randint(10) for i in range(10)]}
print(d1)

d2 = {'c2': [np.random.randint(10, 20) for i in range(5)]}
print(d2)

df1 = pd.DataFrame(data=d1)
df2 = pd.DataFrame(data=d2)

print(df1, df2)

print("1, 2", df1.join(df2))
print("1, 2 right", df1.join(df2, how="right"))
print("1, 2 outer", df1.join(df2, how="outer"))
print("1, 2 inner", df1.join(df2, how="inner"))

print("2, 1",       df2.join(df1))
print("2, 1 right", df2.join(df1, how="right"))
print("2, 1 outer", df2.join(df1, how="outer"))
print("2, 1 inner", df2.join(df1, how="inner"))
