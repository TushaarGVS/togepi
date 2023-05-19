import numpy as np

mha_results = np.zeros((4,4))
nub_heads = np.array([[5,6,8,9],[1,2,3,5],[0,5,6,1]])

f = open("results.txt",'w')
st = "\n\n* Running for "+" heads\n"
f.write(st)
f.write("MultiHeadAttention Results:\n")
f.close()
f = open("resultBIG.txt",'wb')
for line in np.matrix(nub_heads):
	np.savetxt(f, line, fmt='%.2f')
f.close()
