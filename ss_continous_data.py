# Main code to run spectral subtraction algorithm.
# Requires the subroutines in ss_functions.py (spectral subtraction functions).
# Requires a "Parameters file: " to read input file name, input file location, and output file location. 
# parameters.txt file is an example "Parameters file"
# std: [d]egraded stream. Input data. 
# stp: [p]rocessed stream. Output data. 
from obspy import read, Stream
import numpy as np
import os
import sys
import ss_functions as func

filename = input("Parameters file: ")
exec(open(filename).read())
std = read(event_file)
stp = Stream() 
nos = len(std)
std.detrend("linear")
std.detrend("demean")
if not os.path.exists(outpath):
   os.makedirs(outpath)
freqs_array = np.empty((233,)) #233 is the freq number for do_wavelet subroutine in ss_funcitons
for s in range(nos): 
    t = std[s].stats.starttime 
    inc = int((std[s].stats.npts/std[s].stats.sampling_rate)/(tw)) #calculate how many 5 minute time increments are needed for this data 
    windowlength = int(tw*std[s].stats.sampling_rate)
    noise_estimate = np.zeros([233,inc*windowlength])  
    for i in range(0, int(inc)):
        trd = std[s].copy() 
        trd.trim((t + (i*tw)), (t + ((i+1)*tw) - trd.stats.delta)) # stat.delta is necessary to have equal 6000 pts trimmed data. 
        trp = trd.copy() 
        amp_Xd, Xd, freqs, scales, dt, dj = func.do_wavelet(trd) # wavelet transform algorithm called from ss_functions
        amp_Xp, amp_Xna, alpha, rho, a, b = func.do_subtraction(amp_Xd) # spectral subtraction algorithm called from ss_functions
        trp.data = func.do_inv_wavelet(Xd,amp_Xp,trp,scales,dt,dj)# inverse wavelet transform algorithm called from ss_functions
        noise_estimate[:,i*windowlength:i*windowlength+windowlength] = amp_Xna # noise spectrum array
        stp += trp
    stp.merge(method=1)
    stp.sort(['station']) #otherwise BR106 becomes the first trace in the stream
freqs_array = freqs
stp.write(outpath + output_name + '_processed', format="MSEED")