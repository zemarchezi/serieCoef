# encoding: utf-8
# encoding: iso-8859-1
# encoding: win-1252
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.fftpack
import pandas as pd
import glob
from scipy import signal
import time


def normD(a):
    norm = 0
    for i in range(3):
        norm += a[i]*a[i]
    return np.sqrt(norm)

def crossD(a,b):
    cross = [0]*3
    cross[0] = a[1]*b[2]+a[2]*b[1]
    cross[1] = a[2]*b[0]+a[0]*b[2]
    cross[2] = a[0]*b[1]+a[1]*b[2]
    return cross

class DifCoeficient(object):
    """
    Class to read, rotate de coordinate system and separate the Fourier seires coefficients.
    ---
    input:

    filename: string containing the directory of the files (/.../..../directory/*.txt)
    RVal: float indicating the radial distance
    TimeStep: float indicating the simulation time step

    ----
    Modules

    read:
        read all the files in the directory and gives and list of Dataframes, one for each time Step

    rotate_field_fac:
        rotate the fields into Field Alignet Coordinate System

    integPhi:
        Calculate the Fourier Series coefficients
    """

    def __init__(self, filename, RVal, TimeStep):
        self.name = filename # all the directory
        self.radius = RVal # radial distance
        self.tStep = TimeStep # Simulation time step
        self.len = 0 #number of timesteps

    def read(self):
        #read from file
        # faz a leitura de todos os arquivos da pasta
        files = glob.glob(self.name)
        files.sort() # sort the filenames
        self.len = len(files)
        self.inputArray = [0]*self.len
        # armazena em uma lista os data frames para cada passo temporal
        j = 0
        for i in files:
        # data frame temporário contedo todos os dados
            Array = pd.DataFrame(np.loadtxt(i), columns=['x', 'y', 'R', 'phi', 'ex', 'ey', 'ez', 'bx', 'by', 'bz'])
             # lista de dataframes somente com os dados para o R selecionado
            #self.inputArray.append(Array[Array['R'] == self.radius])
            self.inputArray[j] = Array[Array['R'] == self.radius]
            j+=1 

    # convert the coordinate system from gse to field aligned system
    def rotate_field_fac(self):
        '''
        rotate the fields into Field Alignet Coordinate System

        data: pandas dataframe with the columns: 'x', 'y', 'ex', 'ey', 'ez', 'bx', 'by', 'bz'
        '''


        #self.fields = [0]*self.len
        self.fields = []
    # loop in the fataframes od the timesteps
        for k in self.inputArray:
        # temporary data frame for the new rotate fields
        ## mantive os mesmos nomes das variáveis que estavam na rotina do vitor. v1p, v1a, v1r, são e_paralelo, e_azimutal(ephi), e_radial
            #  = pd.DataFrame(np.zeros((len(k['x']), 8)),columns=['bp', 'ba', 'br', 'v1p', 'v1a', 'v1r', 'b_fac', 'b_orig'])
            # Extract the original data components
            x = k['x'].values 
            y = k['y'].values
            z = np.zeros((len(k['x']), 1))
            v1x = k['ex'].values
            v1y = k['ey'].values
            v1z = k['ez'].values
            bx = k['bx'].values
            by = k['by'].values
            bz = k['bz'].values
            tempFields = np.zeros((len(k['x']), 8))
        
            # 'p', 'a', e, 'r', denotam as variáveis na direção paralela, azimutal(phi) e radial
            for i in range(0,len(x)):
                r  = [x[i], y[i], z[i]] / np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])
                # mais uma ves, mantive os mesmos nomes
                ## esses índices referem-se aos vetores unitários nessas direçes.
                ep = [bx[i], by[i], bz[i]] / np.sqrt(bx[i] * bx[i] + by[i] * by[i] + bz[i] * bz[i])
                #ea  = np.cross(ep, r) / np.linalg.norm(np.cross(ep, r))
                #er  = np.cross(ea, ep) / np.linalg.norm(np.cross(ea, ep))
                ea  = crossD(ep, r) / normD(crossD(ep, r))
                er  = crossD(ea, ep) / normD(crossD(ea, ep))
                # apply rotation for B
                tempFields[i][0] = (ep[0] * bx[i]) + (ep[1] * by[i]) + (ep[2] * bz[i])
                tempFields[i][1] = (ea[0] * bx[i]) + (ea[1] * by[i]) + (ea[2] * bz[i])
                tempFields[i][2] = (er[0] * bx[i]) + (er[1] * by[i]) + (er[2] * bz[i])
                # Apply the rotation for vector V1
                tempFields[i][3] = (ep[0] * v1x[i]) + (ep[1] * v1y[i]) + (ep[2] * v1z[i])
                tempFields[i][4] = (ea[0] * v1x[i]) + (ea[1] * v1y[i]) + (ea[2] * v1z[i])
                tempFields[i][5] = (er[0] * v1x[i]) + (er[1] * v1y[i]) + (er[2] * v1z[i])
                # testing whether the rotation is correct
                tempFields[i][6] = normD([tempFields[i][0], tempFields[i][1], tempFields[i][2]])
                tempFields[i][7] = normD([bx[i], by[i], bz[i]])


            tempFieldsDF = pd.DataFrame(tempFields,columns=['bp', 'ba', 'br', 'v1p', 'v1a', 'v1r', 'b_fac', 'b_orig'])
            self.fields.append(tempFieldsDF) #se pegarmos muitos tempos precisamos mudar isso

            
        return (self.fields)


    # Coeficientes de fourier

    def integPhi(self,component,m):
        self.resultA = []
        self.resultB = []

        for k in range(len(self.fields)):
            field = self.fields[k][component].values
            angle = self.inputArray[k]['phi'].values
            angleSize = len(self.inputArray[k]['phi'])
            a = np.zeros(angleSize)
            b = a
            for i in range(angleSize):
                a[i] = field[i] * np.sin(m * angle[i])
                b[i] = field[i] * np.cos(m * angle[i])

            self.resultA.append(integrate.simps(a,angle))
            self.resultB.append(integrate.simps(b,angle))

        return (self.resultA, self.resultB)

###############################################################################################################

class CoeffAnalysis():
    """
    class to organize the fourier coefficients and perform the Power Spectrum Analaysis for each mode number m
    """
    def __init__(self, integral):
        self.integ = integral


    ## Organiza os coeficientes no tempo
    def timeOrg(self):
        self.a = np.zeros((len(self.integ), len(self.integ[0][0])))
        self.b = np.zeros((len(self.integ), len(self.integ[0][0])))
        for m in range(0,len(self.integ)):
            for ti in range(0,len(self.integ[0][0])):
                self.a[m,ti] = self.integ[m][0][ti]
                self.b[m,ti] = self.integ[m][1][ti]

        return (self.a, self.b)

    ## PSD
    def psdM(self):
        fa, ta, Sxxa = [0]*self.a.shape[0], [0]*self.a.shape[0], [0]*self.a.shape[0]
        self.intePot = [0]*self.a.shape[0]
        for i in range(0,self.a.shape[0]):
            fa[i], ta[i], Sxxa[i] = signal.spectrogram(self.b[i,:], 1./30, window='hamming',nperseg=15, nfft=256)
            inPot = np.zeros((Sxxa[i].shape[1]))
            for k in range(0,Sxxa[i].shape[1]):
                inPot[k] = sum(Sxxa[i][k])
            self.intePot[i] = inPot

        return (fa, ta, Sxxa, self.intePot)



    def plots(self):
        fig, axarr = plt.subplots(1, 1, figsize=(15,15))
        plt.pcolormesh(np.log10(self.intePot), cmap='jet')
        axarr.set_title('PSD dif modos')
        axarr.set_xlabel('M Number')
        axarr.set_ylabel('Time')
        axarr.get_xaxis().set_ticks_position('both')
        axarr.get_yaxis().set_ticks_position('both')
        axarr.get_yaxis().set_tick_params(which='both',direction='in')
        axarr.get_xaxis().set_tick_params(which='both',direction='in')
        plt.colorbar(label='PSD')
        plt.savefig('a_psd_teste.png')




######################
# testes
#####################33
filename = 'results_slc/*.txt'
#   sizePhi = 720
#
#
#   iEx = 0
#   iEy = 1
#   iEz = 2
#   iBx = 3
#   iBy = 4
#   iBz = 5
#
#     dC = DifCoeficient(filename)
time0 = time.time()
dC = DifCoeficient(filename, RVal=4.5, TimeStep = 30)
time1 = time.time()
print time1-time0
dC.read() 
time2 = time.time()
print time2-time1
aa = dC.rotate_field_fac()
time3 = time.time()
print time3-time2
bb = []
for m in range(0,45):
    bb.append(dC.integPhi('v1a', m))
time4 = time.time()
print time4-time3
coef = CoeffAnalysis(bb)
time5 = time.time()
print time5-time4
teat = coef.timeOrg()
time6= time.time()
print time6-time5
dd = coef.psdM()
time7 = time.time()
print time7-time6
coef.plots()
time8 = time.time()
print time8-time7

















##test marcos
#tempFields['bp'][i] = (ep[0] * bx[i]) + (ep[1] * by[i]) + (ep[2] * bz[i])
#                tempFields['ba'][i] = (ea[0] * bx[i]) + (ea[1] * by[i]) + (ea[2] * bz[i])
#                tempFields['br'][i] = (er[0] * bx[i]) + (er[1] * by[i]) + (er[2] * bz[i])
#                # Apply the rotation for vector V1
#                tempFields['v1p'][i] = (ep[0] * v1x[i]) + (ep[1] * v1y[i]) + (ep[2] * v1z[i])
#                tempFields['v1a'][i] = (ea[0] * v1x[i]) + (ea[1] * v1y[i]) + (ea[2] * v1z[i])
#                tempFields['v1r'][i] = (er[0] * v1x[i]) + (er[1] * v1y[i]) + (er[2] * v1z[i])
#                # testing whether the rotation is correct
#                tempFields['b_fac'][i] = normD([tempFields['bp'][i], tempFields['ba'][i], tempFields['br'][i]])
#                tempFields['b_orig'][i] = normD([bx[i], by[i], bz[i]])




















































    #
    #
    # def coordTransform(data,phi):
    #   #transformation matrix
    #   A = np.zeros((3,3))
    #
    #   E = np.zeros((len(phi),3))
    #
    #   for i in range(len(phi)):
    #       phi_temp = phi[i]
    #       #A[linha,coluna]
    #       A[0,0] = np.cos(phi_temp)
    #       A[0,1] = np.sin(phi_temp)
    #       A[0,2] = 0.
    #
    #       A[1,0] = -np.sin(phi_temp)
    #       A[1,1] = np.cos(phi_temp)
    #       A[1,2] = 0.
    #
    #       A[2,0] = 0.
    #       A[2,1] = 0.
    #       A[2,2] = 1.
    #
    #       for l in range(3):
    #           temp = 0
    #           for k in range(3):
    #               temp += A[l,k] * data[i,k]
    #           E[i,l] = temp
    #
    #   return E
    #
    #
    #
    # # Fourier series.
    # def Sf(self):
    #     a0 = inte[0][1]
    #     a0 = inte[0]
    #     sum = np.zeros(len(phi))
    #     for j in range(0,len(phi)):
    #         for i in np.arange(1, len(inte)):
    #             sum[j] += (inte[i][0] * np.sin(i*phi[j])) + (inte[i][1] * np.cos(i*phi[j]))
    #
    #     return ((a0/2 + sum)/(np.pi))
    #

#
# def main():
#     inpArray = read(filename)
#     R = 4.5
#     Intt = []
#     Intt2 = []
#     Serie = []
#
#     # data = np.transpose(np.asarray([inpArray[inpArray['R']==R]['ex'].values, inpArray[inpArray['R']==R]['ey'].values, inpArray[inpArray['R']==R]['ez'].values]))
#
#     # Rotate fields into FAC system
#     #
#     fields = rotate_field_fac(inpArray[0])

    # loop para gerar os coeficientes para cada passo temporal
#     for i in inpArray:
#         dda = np.transpose(np.asarray([i[i['R']==R]['ex'].values, i[i['R']==R]['ey'].values, i[i['R']==R]['ez'].values]))
#         Electric = coordTransform(dda, i[i['R']==R]['phi'].values)
#         inte = []
#         serie = []
#         for m in range(0,50):
#             tempInt = integ(Electric[:,1], i[i['R']==R]['phi'].values, m)
#             inte.append(tempInt)
# #            serie.append(Sf(tempInt, i[i['R']==R]['phi'].values))
#
# #        Serie.append(serie)
#         Intt.append(np.asmatrix(inte))
#         Intt2.append(np.asarray(Electric[:,1]))




# #     organiza os coeficientes em uma matrix com valores de m versus tempo
#     aa = np.zeros((50, 50))
#     for k in range(0,len(Intt)):
#         for i in range(0,len(np.asarray(Intt[k][:,0]))):
#             aa[k,i] = np.asarray(Intt[k][i,0])
# ##
#    aa = np.zeros((50, 720))
#    for k in range(0,len(Intt2)):
#        for i in range(0,len(np.asarray(Intt2[k][:]))):
#            aa[k,i] = np.asarray(Intt2[k][i])
#
#
    # plt.pcolormesh(np.log(abs(np.transpose(aa[10:,:]))), cmap='jet')
    # plt.xlabel('time')
    # plt.ylabel('m')
    # plt.colorbar()
    # # plt.show()
    # plt.savefig('a_testeR'+str(R)+'.png')
    #
    # f, t, Sxx = signal.spectrogram(aa[10:,1], 1./30, window='hamming',nperseg=15, nfft=256)
    # f2, t2, Sxx2 = signal.spectrogram(aa[10:,20], 1./30, window='hamming',nperseg=15, nfft=256)
    #
    # fig, axarr = plt.subplots(1, 1)
    # plt.pcolormesh(t,f,np.log10(abs(Sxx)) - np.log10(abs(Sxx2)), cmap='jet')
    # axarr.set_title('PSD dif modos')
    # axarr.set_ylabel('Hz')
    # axarr.set_ylabel('Time')
    # axarr.set_xlim(min(t), max(t))
    # axarr.get_xaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_tick_params(which='both',direction='in')
    # axarr.get_xaxis().set_tick_params(which='both',direction='in')
    # plt.colorbar()
    # plt.savefig('a_psd_teste'+str(R)+'.png')
    #
    # fig, axarr = plt.subplots(1, 1)
    # plt.pcolormesh(t,f,np.log10(abs(Sxx)), cmap='jet')
    # axarr.set_title('PSD m=1')
    # axarr.set_ylabel('Hz')
    # axarr.set_ylabel('Time')
    # axarr.set_xlim(min(t), max(t))
    # axarr.get_xaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_tick_params(which='both',direction='in')
    # axarr.get_xaxis().set_tick_params(which='both',direction='in')
    # plt.colorbar()
    # plt.savefig('a_psd_teste_m_1'+str(R)+'.png')
    #
    # fig, axarr = plt.subplots(1, 1)
    # plt.pcolormesh(t,f,np.log10(abs(Sxx2)), cmap='jet')
    # axarr.set_title('PSD m=20')
    # axarr.set_ylabel('Hz')
    # axarr.set_ylabel('Time')
    # axarr.set_xlim(min(t), max(t))
    # axarr.get_xaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_ticks_position('both')
    # axarr.get_yaxis().set_tick_params(which='both',direction='in')
    # axarr.get_xaxis().set_tick_params(which='both',direction='in')
    # plt.colorbar()
    # plt.savefig('a_psd_teste_m_20'+str(R)+'.png')



#    plt.pcolormesh(aa)
#    plt.colorbar()

#
###
#    data = np.transpose(np.asarray([inpArray[inpArray['R']==R]['ex'].values, inpArray[inpArray['R']==R]['ey'].values, inpArray[inpArray['R']==R]['ez'].values]))
#
#    Electric = coordTransform(data, inpArray[inpArray['R']==R]['phi'].values)
#
##
###    inte  = integ(data[:,1], dim[:,1], 1)
##
##    a0 = integ(data[:,1], dim[:,1], 0)[1]
##
#    inte = []
#    for m in range(0,20):
#        inte.append(integ(Electric[:,1], inpArray[inpArray['R']==R]['phi'].values, m))
#
#    coefA = abs(np.asarray(inte)[:,1])
#    coefB = abs(np.asarray(inte)[:,0])
#    total = 0
#    for i in range(1,len(coefA)):
#        total += coefA[i]+coefB[i]
#
#    print ((coefA[3]+coefB[3]) / total*100)
#
##    inte.append(integ(Electric[:,1], inpArray[inpArray['R']==R]['phi'].values, 1))
#
#    serie  = Sf(inte, inpArray[inpArray['R']==R]['phi'].values)
##    0
#    print (len(serie), serie.size)
##
##    plt.semilogy(a[1], a[0])
##
#
##
#
#    plt.plot(inpArray[inpArray['R']==R]['phi'].values,abs((Electric[:,1] - serie)), label='1')
##
##    plt.plot(inpArray[inpArray['R']==R]['phi'].values, serie, label='2')
#    plt.legend()
#    plt.show()

    # return(inpArray)
#
# if __name__ == '__main__':

#     #
    # inpArray = read(filename)
    # R = 4.5
    # Intt = []
    # Intt2 = []
    # Serie = []

        # data = np.transpose(np.asarray([inpArray[inpArray['R']==R]['ex'].values, inpArray[inpArray['R']==R]['ey'].values, inpArray[inpArray['R']==R]['ez'].values]))

        # Rotate fields into FAC system
    #     #
    # fields = rotate_field_fac(inpArray[0])
    #
    # inpArray = main()

# plt.plot(Serie[22][1])
# plt.plot(Electric[:,1])
