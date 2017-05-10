import numpy as np
from time import time

# Use the largest number of bits here:
b = 28
N = 2**b
x = np.array([ 2*i + (2*i+1)*1j for i in xrange(N)])
print 'Timing b =', b

t1=time()
y = np.fft.fft(x)
t2=time()

print t2-t1, (t2 - t2) / float(N)
