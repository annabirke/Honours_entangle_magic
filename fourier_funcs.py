import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt;
from scipy.linalg import expm
from scipy.signal import find_peaks

def fourier(t_list,signal,xlim,numberpeaks):
    dt=t_list[1]-t_list[0]

    # multiply signal by cosine, with period T=4*signal length, f=2pi/T
    cos=1+np.cos(np.pi/((t_list[-1]-t_list[0])) *t_list)
    signal=signal*cos
    plt.plot(t_list,signal)
    plt.title('Signal*cosine')
    plt.show()
    
    fft=np.fft.fft(signal) # could add dtype float to np array so that ensured real numbers
    freq=np.fft.fftfreq(len(signal),d=dt)
    
    #only positive frequencies
    mask=freq>0
    freq=freq[mask]
    fft=fft[mask]
    power=np.abs(fft)**2
    
    plt.plot(freq,power,'k')
    plt.xlabel('Frequency')
    plt.ylabel('Power of FFT')
    # plt.yscale('log')
    plt.xlim(xlim)
    
    #find peaks
    peaks, _ = find_peaks(power,height=1)

    freqpeak = np.array(freq[peaks])
    powerpeak = np.array(power[peaks])
    indices=np.argsort(powerpeak)[::-1]
    freqpeak = freqpeak[indices]
    powerpeak = powerpeak[indices]

    # numberpeaks=-1 for all except last, or otherwise number of peaks
    for i in range(len(powerpeak[:numberpeaks])):
        plt.axvline(freqpeak[i],ls='--',label=f'f={freqpeak[i]:.3g}')
    plt.legend()
    plt.show()
    return freqpeak, powerpeak

