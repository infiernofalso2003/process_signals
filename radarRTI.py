# ------ process files .wav to imagen sar by python ---------

# import functions 
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib import ticker, cm

# Parameters  to config :

CHANNEL_SIGNAL = 0        # Channel with signal.
CHANNEL_SYNC = 1          # Channel with sync.
NIVEL = 0                 # Level to selecc Pulse sinc.
PRT = 20e-3               # Valor teorico.
LEVEL = 0.5

# ---- Funcion para processar files ---
def loadSignal(pathfile,CHANNEL_SIGNAL=0,CHANNEL_SYNC=1 ):
    # Load data from pathFile
    samplerate, data = wavfile.read(pathfile)    
    # Señal y sicronismo normalizado  :
    signal = (data[:,CHANNEL_SIGNAL]/ np.amax(data[:,CHANNEL_SIGNAL]))
    sync_ = data[:,CHANNEL_SYNC]/np.amax(data[:,CHANNEL_SYNC])
    # Señal de sincronismo logico :
    sync = sync_ > NIVEL
    #sync = sync_
    return signal, sync, samplerate

#  function to proccess signals :
def cutSignals(signal, fs, prt = PRT):
    init = 97
    nSignals = 0
    Npoins = 804 #int(fs*prt)  # 805
    Ntotal = int(len(signal)/Npoins) 
    
    #for index in range(Npoins):
    nSignals = []
    for index in range(Ntotal-1):
        nSignals.append( signal[index*Npoins+init : (index+1)*Npoins + init ] )

#    print ('Size Of all signals :',len(signals))
#    print('Size Of 1 signals : ', Npoins )
    return nSignals

# cut signal to process imag signal from 

def InitIdex(sync):
    index = 0
    if(sync[0] == False) :
        while sync[index] != True : # saco todos los '0'
            index +=1 
    else:
        while sync[index] != False: # saco los '1' que me sobran 
            index +=1
        while sync[index] != True :  # saco los '0 ' que me sobran 
            index +=1

    return index    

def cutSignalsRTI(signals, sync):

    cutSignals = []
    counter = 0 
    index = InitIdex(sync)    # Init index with high puse ('1')

    while index < len(signals) - 800 :
        pulseSignal = []
        # add pulse 1 signas :
        while sync[index] == True :
            pulseSignal.append(signals[index])
            index +=1 
        # add puse 0 signals :
        while sync[index] ==False:
            pulseSignal.append(signals[index])
            index +=1 
        # ----
        if counter == 0 :
            size_of_signal = len(pulseSignal)
            pading = 0 
            counter +=1
        else :
            pading = len(pulseSignal)-size_of_signal

        if pading==0 :  # Caso que tiene la misma dimension 
            cutSignals.append( (pulseSignal) ) # add direct 
        else:
            if pading < 0 :  # Es mas chico y tenes que agregar ceros 
                cutSignals.append(  np.concatenate( ( pulseSignal , np.zeros(abs(pading)) ),axis=0  ))
            else :  # AUX es mas grande que el primero, hay que cortarlo 
                cutSignals.append((  (pulseSignal[0:size_of_signal] ) )) # add direct 

        # add to my list of cut signals 
        # print('size of pulse :', len(pulseSignal))
    return cutSignals  # return all signals

# Remuve clutters :
def clutterRemuve(signals):
    signalsCluterless = []
    signalsCluterless.append(np.array(signals[1])- np.array(signals[0]))
    signalsCluterless.append( np.array(signals[2])- np.array(signals[1]))
    for index in range(2,len(signals)):
        #signalsCluterless.append( np.array(signals[index])-np.array(signals[index-1]) )
       #signalsCluterless.append( np.array(signals[index])-np.array(signals[0]) )
        signalsCluterless.append(2*np.array(1*signals[index])- 1*np.array(signals[index-1])-1*np.array(signals[index-2]) )

    return signalsCluterless

# ---- Media of list of vectors ---- :

def mediaVector(signals):
    media = np.array(signals[0])
    for index in range(1,len(signals)):
        media = media + np.array(signals[index])

    return media/len(signals)

def clutterRemuve2(signals):
    clutterless = []
    media =  np.array(mediaVector(signals))
    #print('len of media', len(media))
    for signal in signals:
        #print('len of signal', len(signal))
        clutterless.append( np.array(signal)-media) 

    return clutterless
# ---- remuve media for list of vector ----

# Get spectro signal :
def spectroSignal(signals):
    return np.fft.fft2(signals)

# dB , 20*log()  :
def dB(signal):
    aux = []
    for value in signal:
        if value > 0.2 :
            aux.append(20*np.log10(value))
        else:
            aux.append(-100)
    return aux

def converter_to_dB(signals) :
    aux = []
    for signal in signals:
        aux.append(dB(signal))
    return aux

def maximous(signals):
    maxValue = max(signals[0])
    for signal in signals :
        if maxValue < max(signal):
            maxValue = max(signal)
        else:
            pass
    return maxValue

def normalize(signals):
    maxValue = maximous(signals)
    return signals/maxValue

def spectroSignal(signals, nfft):
    fftsignals = []
    phaseSignals = []
    for signal in signals:
        aux = ((np.fft.fft(signal, nfft)) )
        #print('Maximo : ', max(aux))
        #aux = aux/max(aux)
        fftsignals.append(( aux[0:int(nfft/2)] ) )
        phaseSignals.append( np.arctan( np.imag(aux[0:int(nfft/2)])/np.real(aux[0:int(nfft/2)]) ))

    #maxValue =  maximous(fftsignals)
    # Normalize 
    #fftsignals = fftsignals/maxValue
        
    return fftsignals, phaseSignals

def spectroSignal2(signals,nfft):
    fftsignals = []
    phaseSignals = []
    for signal in signals:
        aux = ((np.fft.fft(signal, nfft)) )
        #print('Maximo : ', max(aux))
        #aux = aux/max(aux)
        fftsignals.append(abs( aux[0:int(nfft/2)] ) )
        #if np.real(aux[0:int(nfft/2)]) != 0:
        phaseSignals.append( np.arctan( np.imag(aux[0:int(nfft/2)])/np.real(aux[0:int(nfft/2)]) ))
        #else :
        #phaseSignals.append( np.pi/2 )
# **-- tener la fase del sistema nfes que se au :
    maxValue =  maximous(fftsignals)
    # Normalize 
    fftsignals = fftsignals/maxValue
        
    return fftsignals, phaseSignals

# compress in range :
def compressInRange(signals):

    # Compresion con chirp::
    N = 10
    time = np.arange(0,N/2,0.01)
    modulator = 1*np.concatenate(  ( np.exp(6*(-N/2+time)) ,np.exp(-6*(time))   )  , axis=0)
    compressRange = []

    for signal in signals:
        compressRange.append( ( np.convolve(signal,modulator) ))      

    return compressRange

# compress in azimuth :
def compressInAzimuth(signals):

    signals = np.transpose(signals)
    # Compresion con chirp 
    N = 10
    time = np.arange(0,N/2,0.01)
    modulator = 1*np.concatenate(  ( np.exp(10*(-N/2+time)) ,np.exp(-10*(time))   )  , axis=0)
    compressAzimuth = []
    
    for signal in signals:
        compressAzimuth.append( ( np.convolve(signal,modulator) ))      

    return np.transpose(compressAzimuth)

# Function to remuve media froma list of signals :
def remuveMedia(signals):

    medialess = []                         #  append sig() to signals 
    media   = np.mean(signals[0])          # Media reference    
    for signal in signals:
        medialess.append( signal-media )
    
    return medialess
# ---  file name 8 ---
# plot 2-D graphicSs :

def getAngle(signals):
    angles = []
    for signal in signals:
        angles.append( np.arctan( np.imag(signal)/np.real(signal) ))

    return angles

def proccessPhase(phases):

    phaseProcess = []
    for phase in phases:
        aux = []
        for index in range(len(phase)):
            if phase[index] > 0 :
                aux.append(0)
            else:
                aux.append(1)
        phaseProcess.append(aux)

    return phaseProcess

def ifftSignal(signals):
    timesignal = []
    for signal in signals:
        timesignal.append( np.fft.ifft( signal,4096))
    return timesignal


def padding(signals, nzeros):
    singnalPading = []
    for singnal in signals:
        singnalPading.append( np.concatenate( (singnal ,np.zeros(nzeros)) , axis= 0))

    return singnalPading

def makeGraphic(ffsignal):
    print('Make fftsignal graphics')   


def module2D(signals):
    modules = []
    for signal in signals:
        modules.append(abs(signal))
    return modules

def remuveCrossTalk(ffts,cutoff ):
    crossTalkless  = []
    crossTalk = np.concatenate(( np.zeros(cutoff), np.ones(len(ffts[0])-cutoff)) ,axis= 0 ) 
    for fft in ffts:
        crossTalkless.append(fft*crossTalk)

    return crossTalkless
#    COL ,FIL = np.meshgrid( np.arange(0,len(self.imagSarModule),1) ,self.distance)
#    Zmodule = np.transpose(self.imagSarModule)
#    self.PlotWidgetModule.canvas.axes.contourf( self.COL, self.FIL, self.Zmodule  ,cmap= plt.cm.nipy_spectral)
# -- main.py ---
# name file to proccess :
#nameFile = 'sarRTIBarilla\sarRTIBarilla0.wav'    # File path to proccess !
#nameFile = 'sarRTI7\sarRTI0.wav'    # File path to proccess !
#nameFile = 'sarRTIBarillaDist\sarRTIBarillaDist3.wav'    # File path to proccess !
#nameFile = 'sarRTI\sarRTIempty0.wav'
#nameFile = 'sarRTI9\sarRTI3.wav' 

#------ mediciones  18/09/2020 ----
# sarRTIempty
# sarRTIpalovacio
# sarRTIwalkAA

# Loading files :

# ------- file to proccess -------
#nameFile = 'sarRTI7\sarRTI0.wav'    # File path to proccess !
#nameFile = 'sarRTIpaloCorner\sarRTIpaloCorner0.wav'
#nameFile = 'sarRTIpalovacio\sarRTIpalovacio2.wav'    # File path to proccess !

# ------ mediciones  19/09/2020 ----
#nameFile = 'sarRTIBarilla\sarRTIBarilla2.wav'
#nameFile = 'mediciones\_23_09_2020\cuadradoA\cuadradoA0.wav' 
#nameFile = 'mediciones\_22_09_2020\sarRTITCirculo\sarRTITCirculo0.wav'
#nameFile = 'mediciones\_22_09_2020\sarRTITCirculo\sarRTITCirculo0.wav'
# Load signal and sync  signal : 

# ------ compare -------
# cuadrado :
nameFile = 'mediciones\_24_09_2020\CuadradoB\CuadradoB1.wav'
# circulo :
#nameFile = 'mediciones\_24_09_2020\CirculoA\CirculoA1.wav'
# Triangulo :
#nameFile = 'mediciones\_22_09_2020\sarRTITriangulo\sarRTITriangulo1.wav'

print('loading file ...')  # -----

signals, sync, fs = loadSignal(nameFile)

# cuts all pulse signals :
nSignals = cutSignalsRTI(signals, sync)

# Obtengo el modulo y la phase de la fft Original :
#nfft_module, nfft_phase = spectroSignal2(nSignals,4096*2) 

# to remuve media :
#signalMedialess =  remuveMedia(nSignals)

cluterless2 =  clutterRemuve2(nSignals)
print('clutterLess2 done !')

# to remuve clutters 
sigClutterless = clutterRemuve(nSignals)
#sigClutterless = signalMedialess 

#signalMedialess =  padding(signalMedialess,1000)

# sigClutterless = sig
#sigClutterless = sig

# con clutter
fftsClutter, phaseClutter = spectroSignal2(nSignals,4096)



# remuve clutterless
signalCrosstalkless =  remuveCrossTalk(fftsClutter,42)

signalCrosstalkless = module2D(signalCrosstalkless)

# signals fft clutterless :
ffts2, phase = spectroSignal2(cluterless2,4096)

fftsModule2 = module2D(ffts2)

# ------ compress in range and acimut ------------

#fftsModule2 = compressInRange(fftsModule2)


fftsModule2 = compressInAzimuth( fftsModule2)
fftsModule2 = compressInRange(fftsModule2)


fftsClutterModule = module2D(fftsClutter)


dist =  np.arange(0,4096/2,1)*fs/4096
acimut = np.arange(0,len(nSignals),1)


print('len of dist :', len(dist))
print('len of acimut :', len(acimut))
print('len of range :' , len(fftsModule2))
print('len of acimut:',len(fftsModule2[0]))

fftsModule2 = normalize(fftsModule2)
#fftsModule2 =  converter_to_dB(fftsModule2)

plt.figure('Sar Non-Compress clutterless 2 ')
plt.contourf(np.transpose(fftsModule2)  ,  cmap = 'jet')
plt.colorbar()
#plt.ylim(0,250)

plt.figure('Sar Non-Compress with clutter ')
plt.contourf(acimut,dist, np.transpose(fftsClutterModule)  ,  cmap = 'jet')
plt.colorbar()
#plt.ylim(0,250)

plt.figure('Sar remuve clutter a mano ')
plt.contourf(acimut,dist ,np.transpose(signalCrosstalkless)  ,  cmap = 'jet')
plt.colorbar()
#plt.ylim(0,250)

plt.show()

#phaseClutter= clutterRemuve(phase)

#phaseNormal = normalize(phaseClutter)

# Compress in range :
#fftsCompRange = compressInRange(ffts)
#phaseCompRange = compressInRange(phase)
#phaseCompRange = compressInRange(phases)

# Compress in azimuth :
#fftsCompAzunuth = compressInAzimuth(np.transpose(fftsCompRange))
#phaseCompAzimuth =  compressInAzimuth(np.transpose(phaseCompRange))

#phaseCompAzimuth = np.transpose(phaseCompAzimuth)

# normalizo la señal compress in range :
#fftsNormal = normalize(fftsCompRange)


# normaliza the signal compress in azimuth :
#fftsNormalAz = normalize( np.transpose(fftsCompAzunuth))

#timeSignal = ifftSignal(fftsNormalAz)

#fftsClutterNormal =  normalize( fftsClutter )

# paso todo a db
#fftsFilterdB = converter_to_dB(fftsNormal)

# ***** filter f funbtion"****
# plt.figure('chirp to compres ')
# plt.plot(tt,chirp)
# plt.grid(True)
# plt.show()
# compress signal :
#compress = []
# for fft in ffts:
#    compress.append( np.convolve(fft, chirp))
# hamming = np.hamming(6)
# Señal despues del filtro
# *** Proccaess signal from Hamming windows  ***
# fftsdB = []
# for fft in ffts:
#    ff.append(dB(fft))
# for sig in signalsCluterless:
#    fft.append(20*np.log10(abs(np.fft.fft(sig, 2048))))
# signals, sync, fs = loadSignal(nameFile)
# signalsCut = cutSignalsRTI(signals, sync)
# print( 'Init Index : ' ,InitIdex(sync))
# print('len :', len(signalsCut))
# plt.figure('-- Sync Signal ---')
# plt.plot(signalsCut[0])
# plt.grid(True)
# plt.show()
# ff1 = np.fft.fft(signalsCluterless[1], 2048
# -------- proccess all signal --------------
#
#   plt.figure('Signal to process ')
#   plt.plot(fft[1]) 
#   plt.plot(signalsCluterless[1]) 
#   plt.plot(signalsCluterless[2]) 
#   plt.plot(signalsCluterless[3]) 
#   plt.title('fft'e#   plt.grid(True)
#   plt.show()
# --------------------------------------------
# ---------- compress range ------------
#sigCompressRane = compressInRange(ffts)
# --- process file .-wav -----
# ----- base of time and distance , referecnes ----
#time = np.arange(0,((len(ffts))*len(nSignals[0])-1)/fs,len(nSignals[0])/fs)
#time2 = np.arange(0,((len(ffts))*len(nSignals[0])-1)/fs,len(nSignals[0])/fs)
#dist = np.arange(0,len(fftsNormalAz[0]),1)*2.2/566
#time = np.arange(0, len(fftsNormalAz),1)*2/4356

#print('Distancia en rango :',len(dist))
#print('distancia en azimuth :',len(time))

# Levels linear :
#levels_lin = [0 ,0.15, 0.2,0.3,0.4,0.5 ,0.6, 0.7,0.8,0.9,1]
# Levels dB :
#levels_dB = [-60,-40,-20,-10,-6,-3,0]
# levels angule
#levels_ang = np.arange(-3.14,3.14,0.5)

# ---- make graphics -----------
#plt.figure('Sar Non-Compress clutterless ')
#plt.contourf(np.transpose(ffts) , levels = levels_lin, cmap = 'jet')
#plt.colorbar()
#plt.ylabel('Range [m]')
#plt.xlabel('Acimut [m]')
#plt.title('Imagen sar non-compress clutterless')
#plt.ylim(0,200)

# ---- phase ---
#plt.figure('phase sar Non-Compress clutterless')
#plt.contourf( np.transpose(phaseNormal) , cmap = 'jet')
#plt.colorbar()
#plt.ylabel('Distance [m]')
#plt.xlabel('Acimut [puntos]')
#plt.title('phase imagen sar non-compress clutterless')
#plt.ylim(0,200)

#plt.figure('Sar Non-compress')
#plt.contourf( np.transpose(ffts) , levels = levels_lin, cmap = 'jet')
#plt.contourf(time2, dist, np.transpose(fftsFilter) ,levels=levels , cmap = 'inferno')
# cmap = 'jet'
# cmap = 'inferno'
#plt.colorbar()
#plt.ylabel('Distance [cm]')
#plt.xlabel('Time [s]')
#plt.title('Imagen sar non-Compression')
#plt.figure('Sar Compress in range')
#plt.contourf( np.transpose(fftsNormal) , levels = levels_lin, cmap = 'jet')
#plt.contourf(time2, dist, np.transpose(fftsFilter) ,levels=levels , cmap = 'inferno')
# cmap = 'jet'
# cmap = 'inferno'
#plt.colorbar()
#plt.ylabel('Distance [cm]')
#plt.xlabel('Time [s]')
#plt.title('Imagen sar with Range Compression')

# --- make graphics after range process
#plt.figure('Sar Compress In Range and Azimuth ')
#plt.contourf(time,dist,  np.transpose(fftsNormalAz) , levels = levels_lin,cmap = 'jet')
#plt.colorbar()
#plt.ylabel('Range [m]')
#plt.xlabel('Acimuth [m]')
#plt.title('Imagen sar with Range and Azimuth compression')

# ---- phase non compress -------

#plt.figure('Sar Compress phase ')
#plt.contourf( np.transpose(phaseNormal) ,cmap = 'jet')

# plt.contourf(time2, dist, np.transpose(fftsFilter) ,levels=levels , cmap = 'inferno')
# cmap = 'jet'
# cmap = 'inferno'
#plt.colorbar()
#plt.ylabel('Distance [m]')
#plt.xlabel('Time [s]')
#plt.title('Imagen sar phases ')
# ---

#plt.figure('time imag-Sar Non-Compress clutterless ')
#plt.contourf( np.transpose(nSignals), cmap = 'gray')
#plt.colorbar()
#plt.ylabel('Distance [m]')
#plt.xlabel('Acimut [puntos]')
#plt.title('Imagen sar non-compress clutterless')

# grafico original sin comprimir en range and comopres en azicmeiu

#plt.figure('fft signal original non-compress')
#plt.contourf( np.transpose(nfft_module), cmap = 'jet')
#plt.colorbar()
#plt.ylabel('Distance [m]')
#plt.xlabel('Acimut [mm]')
#plt.title('Imagen sar non-compress Original')

#plt.show()
# proces en en gelep,llqo qiue al 
# ----------------------------------------------------------