# ----- ejemplo de filtro hamming -----

import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import kaiserord, lfilter, firwin, freqz
#from matplotlib import pyplot as plt 


fh = 1000
fl = 50 
fs = 20000
NFFT = 4096
N = 100

time = np.arange(0,1,1/fs)

signal = np.sin(2*np.pi*fl*time) + np.sin(2*np.pi*fh*time)
fftsignal = np.fft.fft(signal,NFFT)

freq = np.arange(0,fs/2,fs/NFFT)

fftsignal_module = abs(fftsignal[0: int(NFFT/2)])/(len(signal))

# -- filter --
b =firwin(80, 0.1, window=('kaiser', 8))
# "x" is the time-series data you are filtering
#y =  lfilter(h, 1.0, x)

w, h = freqz(b)

h= 1-abs(h)

signalh = np.convolve(signal,h )



plt.figure('Temporal Signal')
plt.plot(time, signalh)
plt.grid(True)

plt.figure('fft signal')
plt.plot(freq,fftsignal_module)
plt.grid(True)
plt.axis('tight')
plt.text(8000,0.08,'ejemplo')

plt.figure('filter')
plt.plot(w/np.pi, 20 * np.log10(h))
plt.grid(True)
plt.xlabel('freq Normalizada \pi')


plt.show()