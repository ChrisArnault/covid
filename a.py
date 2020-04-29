
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
from scipy.optimize import curve_fit

def model(x, a, b, c, d, e):
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x


def fit_model(x, y):
    fit, params = curve_fit(model, x, y)
    a, b, c, d, e = fit[0], fit[1], fit[2], fit[3], fit[4]
    yfit = model(x, a, b, c, d, e)
    return yfit


def derivative(x, y):
    dy = np.diff(y, 1)
    dx = np.diff(x, 1)
    yfirst = dy / dx
    xfirst = 0.5 * (x[:-1] + x[1:])

    return xfirst, yfirst


file = 'time_series_covid19_confirmed_global.csv'

df = pandas.read_csv(file, sep=',')
# print(df.shape, df.axes, df.values)

t = df.T
# print(t.shape, t.axes, t.values)

p0 = plt.subplot2grid((2, 3), (0, 0))
p1 = plt.subplot2grid((2, 3), (0, 1))
p2 = plt.subplot2grid((2, 3), (0, 2))
p3 = plt.subplot2grid((2, 3), (1, 0))
p4 = plt.subplot2grid((2, 3), (1, 1))
p5 = plt.subplot2grid((2, 3), (1, 2))

regions = ["France", "Italy", "Spain", "US", "Portugal"]
#regions = ["US"]

first_start = 0
for i in range(256):
    s = t[i]
    # print("t=", s.shape, s.ndim, s.size, s.values[1], s.values[4:])
    region = "{}".format(s.values[1])
    if region == "Italy":
        # we consider data only when values reach at least 100
        # and then we re-scale x from the date when reached 100

        first_start = None
        for i, v in enumerate(s.values[4:]):
            if v > 100:
                first_start = i
                break

        break

xlabel = "day from first 100 of Italy"

N = 0
for i in range(256):
    s = t[i]
    # print("t=", s.shape, s.ndim, s.size, s.values[1], s.values[4:])
    region = "{}".format(s.values[1])
    if region not in regions:
        continue
    # print("====", s)

    # we consider data only when values reach at least 100
    # and then we re-scale x from the date when reached 100

    """
    start = None
    for i, v in enumerate(s.values[4:]):
        if v > 100:
            start = i
            break

    if start is None:
        continue
    """
    start = first_start

    try:
        values = np.array([np.float(v) for v in s.values[start:]], np.float)
        print("values shape=", values.shape)

        # we consider data only if the absolute max value is at least 2000
        m = values.max()
        if m < 2000:
            continue

        # raw data
        x = np.array(range(start, start + len(values)))
        print(region, "m=", m, " start=", start, "last=", start + len(values))
        #y = np.array([float(v) / float(m) for v in values])
        y = np.array([float(v) for v in values])

        yfit = fit_model(x, y)

        p0.set(title='absolute data')
        p0.set(xlabel=xlabel)
        p0.plot(yfit, label=region)

        # first derivative
        xfirst, yfirst = derivative(x, y)
        yfit = fit_model(xfirst, yfirst)

        p1.set(title='first derivative')
        p1.set(xlabel=xlabel)
        p1.plot(yfit, label=region)

        # second derivative
        xsecond, ysecond = derivative(xfirst, yfirst)
        yfit = fit_model(xsecond, ysecond)

        p2.set(title='second derivative')
        p2.set(xlabel=xlabel)
        p2.plot(yfit, label=region)

        # raw data
        x = np.array(range(start, start + len(values)))
        print(region, "m=", m, " start=", start, "last=", start + len(values))
        y = np.array([float(v) / float(m) for v in values])

        yfit = fit_model(x, y)

        p3.set(title='scaled data')
        p3.set(xlabel=xlabel)
        p3.plot(yfit, label=region)

        # first derivative
        xfirst, yfirst = derivative(x, y)
        yfit = fit_model(xfirst, yfirst)

        p4.set(title='first derivative')
        p4.set(xlabel=xlabel)
        p4.plot(yfit, label=region)

        # second derivative
        xsecond, ysecond = derivative(xfirst, yfirst)
        yfit = fit_model(xsecond, ysecond)

        p5.set(title='second derivative')
        p5.set(xlabel=xlabel)
        p5.plot(yfit, label=region)

    except:
        print("?")

plt.legend(loc=6)
plt.tight_layout()
plt.show()
