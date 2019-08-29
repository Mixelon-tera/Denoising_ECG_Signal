import matplotlib.pyplot as plt
import csv
from scipy import *
from scipy.optimize import leastsq
import tensorflow as tf
import numpy as np
from scipy.signal import butter, lfilter, freqz
import pylab
import detect_peaks as dp

#Normalization the dataset between 0-1
def normalization(x):
    z=[]
    maxvalue = max(x)
    minvalue = min(x)
    for i in x:
        z.append( (i-minvalue)/(maxvalue-minvalue))
    return z

def zscore(x):
    a = x-np.mean(x)/np.std(x)
    return a

def plotecg(list,time):
    ynorm = list
    x = time
    plt.plot(x, ynorm)
    plt.xlabel('Time (msec)')
    plt.ylabel('MLII (mV)')
    plt.title('ECG Signals')
    plt.legend()
    plt.show()

def multiplotecg(org,noisy,recon,time):
    plt.subplot(1, 1, 1)
    plt.plot(time, org, 'g-', label='Original Signal')
    plt.plot(time, noisy, 'r-', label='Noisy Signal')
    plt.plot(time, recon, 'b-',  label='Denoising Signal')
    plt.xlabel('Time [msec]')
    plt.grid()
    plt.legend()
    #plt.subplots_adjust(hspace=0.35)
    plt.show()

def getdataset(name,dtype):
    mixsize=270
    x=[]
    y1=[]
    combdata = np.zeros((1, mixsize), dtype=np.float64)
    dts = np.zeros((1, mixsize), dtype=np.float64)
    if dtype==0:
        dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/train/' + name
    elif dtype==1:
        dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/test/' + name
    with open(dpath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        #print(spamreader)
        next(spamreader, None)
        next(spamreader, None)
        mix=0
        count=0
        finished=False
        for i in spamreader:
            if count==0:
                sec = int(i[0][3:-5])
                msec = int(i[0][6:-1])
                time = (sec * 1000) + msec
                i[0] = time
                x.append(i[0])
            y1.append(float(i[1]))
            #y2.append(float(i[2]))
            if mix==(mixsize-1):
                sendtime=x
                #print(sendtime)
                ynorm1 = normalization(y1)
                #ynorm1 = y1
                #ynorm2 = normalization(y2)
                #print("ynorm1 size is %d" % len(ynorm1))
                #print("ynorm2 size is %d" % len(ynorm2))
                strd1 = np.asarray(ynorm1)
                #strd2 = np.asarray(ynorm2)
                #print(strd1.shape)
                strd1 = np.reshape(strd1, (1,mixsize))
                #strd2 = np.reshape(strd2, (1, 1440))
                #combdata = np.concatenate((strd1, strd2), axis=0)
                combdata = strd1
                del y1[:]
                #del y2[:]
                finished=True
                count+=1
            mix += 1
            if(finished):
                if count==1:
                    dts = combdata
                else:
                    dts = np.concatenate((dts, combdata), axis=0)
                #print("return chunk file shape = %s" % str(dts.shape))
                mix = 0
                finished=False
    #print("return chunk file shape = %s" %str(dts.shape))
    return x,dts

def getallnoise():
    tera=270
    x = []
    y1 = []
    y2 = []
    combdata = np.zeros((1, tera), dtype=np.float64)
    ns = np.zeros((1, tera), dtype=np.float64)
    ans = np.zeros((1,tera), dtype=np.float64)
    noise_name = ['bw.csv','em.csv', 'ma.csv']
    #noise_name = ['bw.csv']
    nlap=0
    for p in noise_name:
        dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/nstdb/' + p
        with open(dpath, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            next(spamreader, None)
            next(spamreader, None)
            mix = 0
            count = 0
            finished = False
            for i in spamreader:
                if count == 0:
                    sec = int(i[0][3:-5])
                    msec = int(i[0][6:-1])
                    time = (sec * 1000) + msec
                    i[0] = time
                    x.append(i[0])
                y1.append(float(i[1]))
                y2.append(float(i[2]))
                if mix == tera-1:
                    sendtime = x
                    # print(sendtime)
                    ynorm1 = normalization(y1)
                    ynorm2 = normalization(y2)
                    # print("ynorm1 size is %d" % len(ynorm1))
                    # print("ynorm2 size is %d" % len(ynorm2))
                    strd1 = np.asarray(ynorm1)
                    strd2 = np.asarray(ynorm2)
                    # print(strd1.shape)
                    strd1 = np.reshape(strd1, (1, tera))
                    #plotecg(np.reshape(strd1, (1440,)).tolist(), x)
                    strd2 = np.reshape(strd2, (1, tera))
                    #plotecg(np.reshape(strd2, (1440,)).tolist(), x)
                    combdata = np.concatenate((strd1, strd2), axis=0)
                    del y1[:]
                    del y2[:]
                    finished = True
                    count += 1
                mix += 1
                if (finished):
                    if count == 1:
                        ns = combdata
                    else:
                        ns = np.concatenate((ns, combdata), axis=0)
                    mix = 0
                    finished = False

        if nlap==0:
            ans = ns
        else:
            ans = np.concatenate((ans,ns), axis=0)
        nlap +=1
    #print("return chunk file shape = %s" %str(ans.shape))
    return ans

#Generate and plot the signal-dataset with corrupt file
def gendataset(name,type):
    x = []
    y = []
    dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/'+name
    with open(dpath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader, None)
        next(spamreader, None)

        for i in spamreader:
            sec = int(i[0][3:-5])
            msec = int(i[0][6:-1])
            time = (sec*1000)+msec
            i[0] = time
            x.append(i[0])
            y.append(float(i[type]))

    ynorm = normalization(y)

    #a = np.random.uniform(0, 3600,3600*0.3)
    #a = np.floor(a).astype(int)
    #for j in a:
        #ynorm[j]=None
    window_size=360
    for i in range(0,len(ynorm)/window_size):
        plt.plot(x[(window_size*i):(window_size*(i+1))-1],ynorm[(window_size*i):(window_size*(i+1))-1])
        plt.xlabel('Time (msec)')
        plt.ylabel('MLII (mV)')
        plt.title('ECG Signals')
        plt.legend()
        plt.show()

    #return x,ynorm
#Genearate and plot the noise-dataset
def gennoise(name,type):
    x = []
    y = []
    dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/nstdb/' + name
    with open(dpath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader, None)
        next(spamreader, None)
        for i in spamreader:
            sec = int(i[0][3:-5])
            msec = int(i[0][6:-1])
            time = (sec * 1000) + msec
            i[0] = time
            i[type] = float(i[type])
            x.append(i[0])
            y.append(i[type])
    ynorm = normalization(y)
    plt.plot(x, y)
    plt.xlabel('Time (msec)')
    plt.ylabel('%s (mV)' %name)
    plt.title('Noise Signals')
    plt.legend()
    plt.show()

#Conbine the signal-dataset and noise-dataset by corrupting the signal and randomize the signal to append the noise-dataset
def combine(dataname,datatype,noisename,noisetype):
    x = []
    y = []
    n = []
    dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/' + dataname
    npath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/nstdb/' + noisename
    with open(dpath, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        next(datareader, None)
        next(datareader, None)
        for i in datareader:
            sec = int(i[0][3:-5])
            msec = int(i[0][6:-1])
            time = (sec * 1000) + msec
            i[0] = time
            i[datatype] = float(i[datatype])
            x.append(i[0])
            y.append(i[datatype])
        ynorm = normalization(y)
        rc = np.random.uniform(0, 3600, 3600 * 0.3)
        rc = np.floor(rc).astype(int)
        for j in rc:
            ynorm[j] = 0
    with open(npath, 'rb') as csvfile:
        noisereader = csv.reader(csvfile, delimiter=',')
        next(noisereader, None)
        next(noisereader, None)
        for i in noisereader:
            i[noisetype] = float(i[noisetype])
            n.append(i[noisetype])
        nnorm = normalization(n)
        rn = np.random.uniform(0, 3600, 3600 * 0.1)
        rn = np.floor(rn).astype(int)
        for p in rn:
            if(ynorm[p]==None):
                ynorm[p]=nnorm[p]*0.3
            else:
                ynorm[p] += nnorm[p]*0.3
    return x,ynorm

def plotdata(x,ynorm):
    plt.plot(x, ynorm)
    plt.xlabel('Time (msec)')
    plt.ylabel('Normalized-signal (mV)')
    plt.title('Combine Signals')
    plt.legend()
    plt.show()

def plotresult(org_vec,noisy_vec,out_vec,x):
    #original_ecg_data
    plt.plot(x, org_vec)
    plt.xlabel('Time (msec)')
    plt.ylabel('Normalized-signal (mV)')
    plt.title('Original Signals')
    plt.legend()
    #noise_ecg_data
    plt.plot(x, noisy_vec)
    plt.xlabel('Time (msec)')
    plt.ylabel('Normalized-signal (mV)')
    plt.title('Noisy Signals')
    plt.legend()
    #reconstruc_ecg_data
    plt.plot(x, out_vec)
    plt.xlabel('Time (msec)')
    plt.ylabel('Normalized-signal (mV)')
    plt.title('Output Signals')
    plt.legend()
    plt.show()

def gettime():
    x=[]
    count=0
    dpath = '/Users/Mix_Tera/Desktop/ECG_Tera/database/mitdb/train/100.csv'
    with open(dpath, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        next(spamreader, None)
        next(spamreader, None)
        for i in spamreader:
            if count<1620:
                sec = int(i[0][3:-5])
                msec = int(i[0][6:-1])
                time = (sec * 1000) + msec
                i[0] = time
                x.append(i[0])
            count +=1
    return x

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def lowfilter(ecgdata):
    # Filter requirements.
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 10#3.667  # desired cutoff frequency of the filter, Hz

    # Get the filter coefficients so we can check its frequency response.
    b, a = butter_lowpass(cutoff, fs, order)

    #ntime , data = getdataset('213.csv',1)
    #data = data[0:6]
    #data = np.reshape(data, (1620,)).tolist()
    #time = gettime()
    y = butter_lowpass_filter(ecgdata, cutoff, fs, order)
    return y
    #plt.plot(time, data, 'b-', label='data')
    #plt.plot(time, y, 'g-', linewidth=2, label='filtered data')
    #plt.xlabel('Time [sec]')
    #plt.grid()
    #plt.legend()
    #plt.subplots_adjust(hspace=0.35)
    #plt.show()

def getpeak(ecgdata):
    indexes = dp.detect_peaks(normalization(np.reshape(ecgdata, (270,)).tolist()), mph=0.9, mpd=100, threshold=0, edge='rising')
    return ((indexes*75)/27)

def cooldown(ecgdata,type):
    h_peak = dp.detect_peaks(normalization(np.reshape(ecgdata, (270,)).tolist()), mph=0.8, mpd=100, threshold=0,
                             edge='rising')
    listpeak = []
    for peak in h_peak:
        for v in range(0, 10):
            listpeak.append(peak - v)
            listpeak.append(peak + v)
    listpeak = list(set(listpeak))
    #print('High Peaks are: %s' % ((h_peak * 75) / 27))
    outvec2 = np.reshape(ecgdata, (270,)).tolist()
    for r in range(len(outvec2)):
        if (r not in listpeak):
            if(type==0):
                outvec2[r] = outvec2[r] * 1
            elif(type==1):
                outvec2[r] = outvec2[r] * 0.2
    #outvec2 = lowfilter(outvec2)
    return outvec2

#gendataset("213.csv",1)
#gennoise("ma.csv",2)
#combine("100.csv",1,"bw.csv",2)

#getallnoise()
'''
x=[0.86,0.88,0.81,0.85,0.89]
y=[-5.98,0.14,17.45,29.11,30.22]
tera=[]
for l in range(len(x)):
    tera.append(x[l]+(0.3*y[l]))
mix=[]
#z = normalization(y)
x = normalization(x)
print(x)
#print(z)
#for e in range(len(x)):
    #mix.append((z[e]*0.3)+x[e])
#print(normalization(mix))
#print(normalization(tera))
'''
