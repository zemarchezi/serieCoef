import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.fftpack
import pandas as pd
import glob

def read(name):
	#read from file
    files = glob.glob(name)
#    for i in files:
    
        
    inputArray = pd.DataFrame(np.loadtxt(name), columns=['x', 'y', 'R', 'phi', 'ex', 'ey', 'ez', 'bx', 'by', 'bz'])
    f = open(name,'r')
    
    
#	lines = f.readlines()
#
#	#split into columns 
#	lineSize = len(lines)
#
#	if lineSize!=sizePhi:
#		#deveria retornar algum erro
#		exit()
#
#	xTemp = np.zeros((lineSize,2))
#	rTemp = xTemp
#
#	inputArray = np.zeros((lineSize,6))
#
#	i = 0
#	for x in lines:
#		for j in range(2):
#			xTemp[i,j] = float(x.split()[j])
#
#		for j in range(2,4):
#			rTemp[i,j-2] = float(x.split()[j])
#
#		for j in range(4,10):
#			inputArray[i,j-4] = float(x.split()[j])
#		i+=1
#
#	f.close()

    return inputArray

def coordTransform(data,phi):
	#transformation matrix
	A = np.zeros((3,3))

	E = np.zeros((len(phi),3))

	for i in range(len(phi)):
		phi_temp = phi[i]
		#A[linha,coluna]
		A[0,0] = np.cos(phi_temp)
		A[0,1] = np.sin(phi_temp)
		A[0,2] = 0.

		A[1,0] = -np.sin(phi_temp)
		A[1,1] = np.cos(phi_temp)
		A[1,2] = 0.

		A[2,0] = 0.
		A[2,1] = 0.
		A[2,2] = 1.
        
		for l in range(3):
			temp = 0
			for k in range(3):
				temp += A[l,k] * data[i,k]
			E[i,l] = temp

	return E


def integ(y,x,m):
    a = [0]*len(y)
    b = [0]*len(y)
    for i in range(sizePhi):
        a[i] = y[i] * np.sin(m * x[i])
        b[i] = y[i] * np.cos(m * x[i])
        	
    resultA = integrate.simps(a,x)
    resultB = integrate.simps(b,x)
    
    return ([resultA, resultB])

# Fourier series.
def Sf(inte, phi):
    a0 = inte[0][1]
    sum = np.zeros(len(phi))
    for j in range(0,len(phi)):      
        for i in np.arange(1, len(inte)):
            sum[j] += (inte[i][0] * np.sin(i*phi[j])) + (inte[i][1] * np.cos(i*phi[j]))
    
    return ((a0/2 + sum)/(np.pi))

#
#def psd(xx, dt):
#    mean_removed = np.ones_like(xx) * np.mean(xx)
#    xx = xx - mean_removed
#    TT = len(xx)
#    fny = 0.5 * (1 / dt)
#    N_fft = 50000        # Number of bins (chooses granularity)
#    Fs = 1.0
#
##    dt = (tt[1] - tt[0]).total_seconds() # passo 1 segundo
##    tt = ss[0].index # array tempo
#    
#    # Compute the fft.
#    yf = scipy.fftpack.fft(xx,n=N_fft)
#    xf = (np.arange(0,fny,fny/N_fft))
#    psd = (np.abs(yf) ** 2)
#    
#    return ([psd, xf])

def main():
    inpArray = read(filename)
    
    R = 8.0
#
    data = np.transpose(np.asarray([inpArray[inpArray['R']==R]['ex'].values, inpArray[inpArray['R']==R]['ey'].values, inpArray[inpArray['R']==R]['ez'].values]))
    
    Electric = coordTransform(data, inpArray[inpArray['R']==R]['phi'].values)
    
#    
##    inte  = integ(data[:,1], dim[:,1], 1)
#    
    a0 = integ(data[:,1], dim[:,1], 0)[1]
#    
    inte = []
    for m in range(0,1):
        inte.append(integ(Electric[:,1], inpArray[inpArray['R']==R]['phi'].values, m))
        
#    inte.append(integ(Electric[:,1], inpArray[inpArray['R']==R]['phi'].values, 1))

    serie  = Sf(inte, inpArray[inpArray['R']==R]['phi'].values)
#    0
    print (len(serie), serie.size)
#    
#    plt.semilogy(a[1], a[0])
#    

#   
    
    plt.plot(inpArray[inpArray['R']==R]['phi'].values,abs((Electric[:,1] - serie)), label='1')
#    
#    plt.plot(inpArray[inpArray['R']==R]['phi'].values, serie, label='2')
    plt.legend()
#    plt.show()
    
    return(inte)
	
if __name__ == '__main__':
	filename = 'results_slc 2/file_0001.txt'
	sizePhi = 720


	iEx = 0
	iEy = 1
	iEz = 2
	iBx = 3
	iBy = 4
	iBz = 5

	intwe = main()
	
