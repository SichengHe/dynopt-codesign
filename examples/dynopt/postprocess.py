from pyoptsparse import OPT, Optimization, History
import numpy as np

filename = "code\dynOpt\ipopt_CCD.hst"
hist = History(filename, flag="r")
xdict = hist.getValues(names="xvars", major=True)

# filename = "opt_ae_hist.dat"
filename = "opt_hist.dat"
np.savetxt(filename, xdict["xvars"])