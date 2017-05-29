import soundfile as sf
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

#read data, the type of data is a 1-D np.ndarray
data1, fs1 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig1.wav')
data2, fs2 = sf.read('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\a_sig2.wav')

#L is the number of independent sources a priori
L = 2;
#this sets the random seed to a fixed number.
np.random.seed(10)

#print(data1)

#randomly initialize the mixing matrix A
#each entry is from uniform[0,1), 
A = np.random.rand(2,2)
#stack the two data arrays together as the source signals
#the shape of S is (2,Ns)
S = np.array((data1,data2))

#dimension is the number of rows. 
dimension = len(S)
#the number of data points. Also the number of columns.
Ns = len(data1)
#V is the observed signal mixture.
V = np.dot(A,S)
#write the mixed multichannel audio file. data should be in the shape(NSample,NChannel)
scipy.io.wavfile.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\mix.wav', fs1, V.T)
#epsilon is the threshold value
epsilon = 1e-7;
#B is the store place for estimated demixing w vectors
B = np.zeros((L,dimension))
#randomly intialize demixing matrix W
W = np.random.rand(L,dimension)
#iterations can record the number of iterations to find a demixing vector w
iterations = np.zeros((1,dimension))

#Remove mean
#To take the mean of each row, choose axis = 1
meanValue = np.mean(V, axis = 1)
#This changes meanValue from 1d to 2d, now a column vector with size dimension*1
meanValue = np.reshape(meanValue,(len(meanValue),1))
#This creates an array full of ones with the same length as the column number of V
oneArray = np.ones((1,Ns))
#This creates a matrix full of mean values for each row
meanMatrix = np.dot(meanValue,oneArray)
#This gives V zero mean
V = V - meanMatrix

#whitening
#this computes the covariance matrix of V. Each row should be a variable and each column should be an observation.
covMatrix = np.cov(V)
#this gets the svd form of the covMatrix.
P,d,Qt = np.linalg.svd(covMatrix, full_matrices=False)
Q = Qt.T
#this gets the first L entries
d = d[:L]
D = np.diag(d)
#this gets the first L columns of singular (eigen) vectors
E = P[:,:L]
#this computes the whitening matrix D^(-1/2)*E.T
whiteningMatrix = np.dot(np.linalg.inv(np.sqrt(D)),E.T)
#X is the whitened signal matrix
X = np.dot(whiteningMatrix,V)

#Perform ICA
for round in range(L):
	#w is a column of W
	w = W[:,round].reshape(L,1)
	#this represents the previous w during the 1000 iterations below
	wOld = np.zeros((L,1))
	
	for i in range(1,1001):
		#Orthogonalizing projection
		w = w - np.dot(np.dot(B,B.T),w) 
		#normalize w
		w = np.divide(w,np.linalg.norm(w))

		# If it is converged, quit
		if np.linalg.norm(w-wOld) < epsilon or np.linalg.norm(w+wOld) < epsilon:
			#to convert w from shape(2,1) to (2,)
			B[:,round] = w.reshape(L)
			W[round,:] = np.dot(w.T,whiteningMatrix)
			break
		#update wOld
		wOld = w
		hypTan = np.tanh(np.dot(X.T,w))
		w = np.divide((np.dot(X,hypTan) - np.dot(np.sum(1 - np.square(hypTan)).T, w)), Ns)
		w = np.divide(w,np.linalg.norm(w))
	iterations[0,round] = i

#Add back mean
#W = [[27.2614, -33.2637], [33.6309, -1.4574]]
Y = np.dot(W,V) + np.dot(np.dot(W,meanValue),np.ones((1,Ns)))
#Y = np.divide(Y,np.amax(np.absolute(Y)))
#plt.plot(Y[0,:5*fs1])
#print(Y)
#print(S)
#print(V)
#print(A)
#print(W)
#print(np.dot(W,A))
sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\new1.wav', Y[0,:], fs1)
sf.write('E:\\Courses\\Comp489\\ICA\\ICAFast\\Data\\new2.wav', Y[1,:], fs1)
#plt.show()

