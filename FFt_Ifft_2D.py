# ejemplo de fft de 2 D

import numpy as np
from matplotlib import pyplot as plt

# ---- fft 2 d ep ar ale caopr  ----
def fft2D(signals, nfft):
    
    nfftsignal_Module = []
    nfftsignal_Phase = []

    for signal in signals:
        aux = np.fft.fft(signal,nfft)
        nfftsignal_Module.append(aux[0:int(nfft/2)])
        nfftsignal_Phase.append(np.angle(aux[0: int(nfft/2)]) )
    
    return nfftsignal_Module, nfftsignal_Phase

# ----- nfft 2D ---------

def nfft(signals,nfft_):
    nfftsignal = []
    for signal in signals:
        aux = np.fft.ifft(signal,nfft_)
        nfftsignal.append(aux[0:int(nfft_/2)])
    
    return nfftsignal

def nfft2D(signals, nfft_) :
    # fft in range
    nfft_range = nfft(signals, nfft_)
    # fft in azimuth
    nfft_acimut = nfft( np.transpose(nfft_range),nfft_)

    return np.transpose(nfft_acimut)

# ----- infft 2D ---------

def ifft(signals, nfft_):
    nifftsignal = []
    for signal in signals:
        aux = np.fft.ifft(signal,nfft_)
        nifftsignal.append(aux[0:int(nfft_/2)])
    
    return nifftsignal

def nifft2D(signals, nfft_):
    # ifft in range 
    nifft_range = ifft(signals, nfft_)
    # ifft in acimut  
    nifft_acimut = ifft( np.transpose(nifft_range), nfft_)
   
    return np.transpose(nifft_acimut)

# ------------------------------

def module2D(signals):
    module = []
    for signal in signals:
        module.append( abs(signal))
    
    return module


def filterHamming(ffts):
    N = len(ffts[0])
    n = np.arange(0,N, 1)
    H = 0.5 + 0.5*np.cos(2*np.pi*(n-N/2)/N)
    
    hffts=[]
    for fft in ffts:
        hffts.append( fft*H )

    return hffts

def setRangeAcimut(signals, range_, acimut_, offset_):
    rma = []

    for index in range(acimut_):
        rma.append( signals[index][offset_:range_+offset_+10] )

    return rma

# ----- parameters ------

fs = 4e3
f0 = 200
NFFT = 4096

# -----------------------

time =  np.arange(0,0.1,1/fs)
signal0 = np.sin(2*np.pi*f0*1*time)
signal1 = np.sin(2*np.pi*f0*2*time)
signal2 = np.sin(2*np.pi*f0*3*time)
signal3 = np.sin(2*np.pi*f0*4*time)

nsignal = []

for index in range(10):
    nsignal.append(signal0)
for index in range(1):
    nsignal.append(signal3)

for index in range(10):
    nsignal.append(signal1)

for index in range(10):
    nsignal.append(signal2)
for index in range(10):
    nsignal.append(signal3)


freq = np.arange(0,fs/2,fs/NFFT)

imagsar = nfft2D(nsignal,NFFT)

imagsar = module2D(imagsar)

nfft_range = nfft(nsignal,NFFT )
nfft_range = module2D(nfft_range)

#nfft_range = nfft2D(nsignal,nfft)

#nfft_range_module = module2D(nfft_range)


#nsignal = []


# aplico el filtro hamming en spectro de frecuencias
#nfft_range = filterHamming(nfft_range)

#nfft_acimut = nfft2D( np.transpose(nfft_range) , nfft )

#nfftModule = module2D(nfft_acimut)

# Ifft en acimut

#infft_acimut = nifft2D(nfft_acimut,nfft) 
#infft_range = nifft2D( np.transpose(infft_acimut), nfft)

#infftModule = module2D(infft_acimut)

# - free memory ---
#nfft_range = []
#nfft_acimut = []
#infft_acimut = []

# set Range and acimut values :
#imagsar = setRangeAcimut(infft_range, 840, 50, 5)
#imagsar = module2D(imagsar)


# -----  filtro de hamming --------

#fftsignal = np.fft.fft2(nsignal)
#fftsignal_module = abs(fftsignal[0:len(fftsignal)])
#fftsignal_phase = np.angle(fftsignal[0:len(fftsignal)])


#print('size of range :', len(nfftModule))
#print('size of acimut :',len(nfftModule[0]))
#print('size of range :', len(freq))
#print('size of acimut :' , len(acimut))

# graphics :
#plt.figure('nfft range and acimut ')
#plt.contourf( infftModule , cmap = 'jet')
#plt.colorbar()

plt.figure('fft range')
plt.contourf(np.transpose( nfft_range), cmap='jet')
plt.colorbar()
#plt.grid(True)
#plt.xlim(0,fs/2)

#plt.figure('phase fft signal')
#plt.plot(freq,fftsignal_phase)
#plt.grid(True)
#plt.xlim(0,fs/2)


#plt.figure('nsignals echo')
#plt.contourf( np.transpose(nsignal), cmap='jet')
#plt.colorbar()

plt.figure('imag sar')
plt.contourf(np.transpose(imagsar) , cmap='jet')
plt.colorbar()


plt.figure('nsignal orig')
plt.contourf(np.transpose(nsignal) , cmap='jet')
plt.colorbar()


plt.show()