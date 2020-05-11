
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import DayLocator, MonthLocator, DateFormatter, drange
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas
from scipy.optimize import curve_fit

def model1(x, a, b, c, d, e):
    return a + b*x + c*x*x + d*x*x*x + e*x*x*x*x


def model2(x, scale, mean, sigma):
    try:
        # a = 1.0/(sigma*np.sqrt(2 * np.pi))
        x2 = (x - mean)/sigma
        x3 = np.power(x2, 2)
        y = scale*np.exp(-x3/2)
        return y
    except:
        print("???")



def fit_model1(x, y):
    fit, params = curve_fit(model1, x, y)
    a, b, c, d, e = fit[0], fit[1], fit[2], fit[3], fit[4]
    yfit = model1(x, a, b, c, d, e)
    return yfit


def fit_model2(x, y):
    scale = max(y)
    mean = 100.0
    sigma = 30.0
    fit, params = curve_fit(model2, x, y, scale, mean, sigma)
    scale, mean, sigma = fit[0], fit[1], fit[2]
    yfit = model2(x, scale, mean, sigma)
    return yfit


def fit_model(x, y):
    return fit_model1(x, y)


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
    axes = s.axes
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

def date_split(d):
    return [int(de) for de in d.split('/')]

def date_cnv(d):
    ds = date_split(d)
    r = datetime.datetime(2000+ds[2], ds[0], ds[1])
    return r

def plot_data(title, region, axe, dates, y):
    axe.set(title=title)
    axe.set(xlabel="day from first 100 of Italy")
    axe.xaxis.set_major_locator(MonthLocator())

    locator = DayLocator(range(0, 31, 1))
    locator.MAXTICKS = 400
    axe.xaxis.set_minor_locator(locator)

    axe.xaxis.set_major_formatter(DateFormatter('%B'))
    axe.fmt_xdata = DateFormatter('%B')
    axe.plot_date(dates, y, label=region, ls='-', marker='.', ms=1)
    # axe.plot_date(dates, y, label="aaa")


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

    axes = s.axes[0]
    all_dates = [date_cnv(d) for d in axes[start:]]
    delta = datetime.timedelta(days=1)
    dates = drange(all_dates[0], all_dates[-1]+delta, delta)

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
        # y = np.array([float(v) / float(m) for v in values])
        vv = [float(v) for v in values]
        y = np.array(vv)
        scale = np.max(y)
        mean = 110
        sigma = 20

        # y = model2(x, scale, mean, sigma)
        # plot_data("absolute data", region, p0, dates, y)
        # plt.show()
        # exit()
        yfit = fit_model(x, y)

        plot_data("absolute data", region, p0, dates, yfit)

        # first derivative
        xfirst, yfirst = derivative(x, y)
        yfit = fit_model(xfirst, yfirst)
        plot_data('first derivative', region, p1, dates[:-1], yfit)

        # second derivative
        xsecond, ysecond = derivative(xfirst, yfirst)
        yfit = fit_model(xsecond, ysecond)
        plot_data('second derivative', region, p2, dates[:-2], yfit)

        # raw data
        x = np.array(range(start, start + len(values)))
        print(region, "m=", m, " start=", start, "last=", start + len(values))
        y = np.array([float(v) / float(m) for v in values])

        yfit = fit_model(x, y)
        plot_data('scaled data', region, p3, dates, yfit)

        # first derivative
        xfirst, yfirst = derivative(x, y)
        yfit = fit_model(xfirst, yfirst)
        plot_data('first derivative', region, p4, dates[:-1], yfit)

        # second derivative
        xsecond, ysecond = derivative(xfirst, yfirst)
        yfit = fit_model(xsecond, ysecond)
        plot_data('second derivative', region, p5, dates[:-2], yfit)

    except:
        print("?")

plt.legend(loc=6)
plt.tight_layout()
plt.show()
