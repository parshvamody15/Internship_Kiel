# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Defining all functions used in the code
def read_PETSc_vec(file):
    f = open(file, "rb")
    np.fromfile(f, dtype=">i4", count=1)
    nvec, = np.fromfile(f, dtype=">i4", count=1)
    v = np.fromfile(f, dtype=">f8", count=nvec)
    f.close()
    return v
 
def read_PETSc_matrix(file):
    f = open(file,'rb')
    np.fromfile(f, dtype=">i4", count=1)     # PETSc matrix cookie
    nrow   = int(np.fromfile(f, dtype=">i4", count=1))      # number of rows
    ncol   = int(np.fromfile(f, dtype=">i4", count=1))      # number of columns
    nnzmat, = np.fromfile(f, dtype=">i4", count=1)      # number of nonzeros
    nnzrow = np.fromfile(f, dtype=">i4", count=nrow)   # number of nonzeros per row
    colind = np.fromfile(f, dtype=">i4", count=nnzmat) # column indices
    aij    = np.fromfile(f, dtype=">f8", count=nnzmat) # nonzeros
    f.close()
    A = np.zeros((nrow, ncol))
    start = 0
    for i in range(nrow):
         if nnzrow[i] != 0:
             end = start + nnzrow[i]
             A[i,colind[start:end]] = aij[start:end]
             start = end
    return A 

def reshape_vector_to_3d(landSeaMask, v):
    # landseamask has shape latitude x longitude
    # we want the output to have longitude x latitude
    landSeaMask = np.transpose(landSeaMask)
    n1, n2 = np.shape(landSeaMask)
    n3 = int(np.amax(landSeaMask))
    v3d = np.nan * np.ones((n1,n2,n3))
    offset = 0
    # v is stored latitude x longitude
    for j in range(n2):
         for i in range(n1):
             if landSeaMask[i,j] != 0:
                 n = int(landSeaMask[i,j])
                 v3d[i,j,:n] = v[offset:offset + n]
                 offset = offset + n
    return v3d, n1, n2, n3

# Reading the N output.petsc files from the folders given by taking required input from the user.
par_list=[]
par_list.append(read_PETSc_vec('/Users/parshvamody/Desktop/Climate_Model/Python/N_100/Parameter_00000/N_output.petsc'))
print('00000: Reference')
print(par_list[0])
print('Size: '+str(np.shape(par_list[0])))
print('                 ')
filenames=[]
inp1=str(input('Enter starting file number: '))
inp2=str(input('Enter ending file number (not included): '))
print('                 ')
r=(int(inp2)-int(inp1)+1)
print(r)
for i in range(int(inp1),int(inp2)):
    i=str(i)
    i=i.zfill(2)
    filenames.append('/Users/parshvamody/Desktop/Climate_Model/Python/N_100/Parameter_000'+i+'/N_output.petsc')
for j in range(0,r-1):
    par_list.append(read_PETSc_vec(filenames[j]))
    print('000'+str(j+int(inp1))+':')
    print(par_list[j+1])
    print('Size: '+str(np.shape(par_list[j])))
    print('                 ')
print('Size of the entire list is: '+str(np.shape(par_list)))
print('                 ')

# Reading in the landSeaMask.petsc file
print('Land Sea Mask file:')
landSeaMask=read_PETSc_matrix('/Users/parshvamody/Desktop/Climate_Model/Python/N/landSeaMask.petsc')
print(landSeaMask)
print('                 ')
print('Shape of landSeaMask file is: '+str(np.shape(landSeaMask)))

# Plotting the land sea mask file
plt.figure(figsize=(13,7))
plt.imshow(np.transpose(np.flipud(np.transpose(landSeaMask))), cmap='hot')
plt.colorbar()
plt.title('Land Sea Mask Representation')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()
plt.show()

# Reshaping all data sets into a 3D vector
reshape_vector=[]
for i in range(0,r):
    re_vector, v1, v2, v3=reshape_vector_to_3d(landSeaMask, par_list[i])
    reshape_vector.append(re_vector)
print('                 ')

# Reducing the reshaped data to the upper surface
upper_layer=[]
for i in range(0,r):
    upper_layer_data=reshape_vector[i][:,:,0]
    upper_layer.append(upper_layer_data)
print('Shape of upper layer data is: '+str(np.shape(upper_layer)))
print('                 ')

# Transforming the data again into a 1D vector
upper_layer_1d=[]
for i in range(0,r):
    upper_layer_1d_data=upper_layer[i].flatten()
    upper_layer_1d.append(upper_layer_1d_data)
print('Shape of 1D upper layer data is:'+str(np.shape(upper_layer_1d)))
print('                 ')

# Putting all datasets into 1 matrix except for dataset 00000
print('Matrix with all datasets:')
upper_mat=np.vstack((upper_layer_1d[1:r]))
print(upper_mat)
print('Shape of matrix with all datasets: '+str(np.shape(upper_mat)))
print('                 ')

# Replacing NaN values by 0 for the matrix
print('Number of NaN values in the matrix: '+str(sum(sum(np.isnan(upper_mat)))))
print('                 ')
upper_mat[np.isnan(upper_mat)]=0
print('Matrix after replacing all NaN values with 0:')
print(upper_mat)
print('                 ')

# Replacing NaN values by 0 for the file 00000
print('Number of NaN values in 00000: '+str(sum(np.isnan(upper_layer_1d[0]))))
print('                 ')
upper_layer_1d[0][np.isnan(upper_layer_1d[0])]=0
print('00000 file after replacing all NaN values with 0:')
print(upper_layer_1d[0])
print('                 ')

# Taking average of every datapoint present in the matrix and subtracting it from every datapoint in the matrix
average=np.mean(upper_mat)
res_mat=upper_mat-average
print('Resultant matrix after taking difference from point wise average:')
print(res_mat)
print('Shape of resultant matrix: '+str(np.shape(res_mat)))


# Singular value decomposition of this matrix
u, s, vh = np.linalg.svd(res_mat, full_matrices=False)
# u: Unitary matrices, s: The singular value for every matrix, in descending order, v: Unitary matrices.
print('                 ')

# Printing shape of SVD outputs
print('Size SVD output is:')
print('u shape: '+str(np.shape(u)))
print('s shape: '+str(np.shape(s)))
print('vh shape: '+str(np.shape(vh)))
print('                 ')

# Printing SVD outputs
print('SVD outputs are:')
print('u: '+str(u))
print('                 ')
print('s: '+str(s))
print('                 ')
print('vh: '+str(vh))
print('                 ')

# Printintg curve for singular values (s)
plt.figure(figsize=(13,7))
plt.plot(s)
plt.title('Curve for values of s')
plt.ylabel('Value')
plt.xlabel('Index')
plt.grid()
plt.show()

# Calculating number of singular values needed to represent the entire data
s_num=s_denom=0
inp3=int(input('Enter number of singular values required: '))
for i in range(0,inp3):
    s_num+=(s[i]*s[i])
    num_singular_values=i+1
for i in range(0,r-1):
    s_denom+=(s[i]*s[i])
print('                 ')
print('Percentage of data represented: '+str((s_num/s_denom)*100))
print('                 ')

for i in range(0,inp3):
    a=np.reshape(vh[i],(128,64))
    a[a==0]=np.nan
    plt.figure(figsize=(13,7))
    plt.imshow(np.transpose(np.flipud(a)),cmap='jet')
    plt.colorbar()
    plt.title('Graph for reshaped vh['+str(i)+'] : singular vector')
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.grid()
    plt.show()

# Reconstruct the data set in the folder Parameter 00000 with the calculated eigenvalues. 
# (See page 44 in the lecture Methods of Data Analysis.)

# Reshape the reconstructed and the reference data into a two-dimensional field

# Plot both two-dimensional fields and compare them. (For that step, you should replace zero with NaN again.)
# generate random integer values

u_trunc=u[:,:num_singular_values]
s_trunc=np.diag(s[:num_singular_values])
vh_trunc=vh[:num_singular_values,:]
print('Reshaped u_trunc size: '+str(np.shape(u_trunc)))
print('                 ')
print('Reshaped s_trunc size: '+str(np.shape(s_trunc)))
print('                 ')
print('Reshaped vh_trunc size: '+str(np.shape(vh_trunc)))
print('                 ')

reconstructed_data=u_trunc@s_trunc@vh_trunc
reconstructed_data[reconstructed_data == 0] = np.nan
"""
# Reconstructed data after multiplying all truncated values, output is very small hence commented out
plt.figure(figsize=(13,7))
plt.imshow(reconstructed_data,cmap='jet')
plt.colorbar()
plt.title('Reconstructed data after using '+str(num_singular_values)+' singular values from SVD')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()
plt.show()
print('Reconstructed Data: '+str(reconstructed_data))
print('                 ')
print('Shape of reconstructed data: '+str(np.shape(reconstructed_data)))
"""

plt.figure(figsize=(13,7))
# First subplot
plt.subplot(1, 2, 1)
upper_new = np.reshape(upper_mat[0], (128, 64))
plt.imshow(np.transpose(np.flipud(upper_new)), cmap='jet')
plt.colorbar()
plt.title('File 000' + str(inp1) + ' after replacing NaN values with 0')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()

# Second subplot
plt.subplot(1, 2, 2)
new = np.reshape(reconstructed_data[0], (128, 64))
plt.imshow(np.transpose(np.flipud(new)), cmap='jet')
plt.colorbar()
plt.title('Reshaped file 000' + str(inp1))
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()
plt.tight_layout()
plt.show()

"""
plt.figure(figsize=(13,7))
plt.subplot(1,2,1)
meh=np.reshape(upper_layer_1d[0],(128,64))
plt.imshow(np.transpose(np.flipud(meh)), cmap='jet')
plt.colorbar()
plt.title('00000')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()

meh2=upper_layer_1d[0]*s[:1]
plt.subplot(1,2,2)
meh3=np.reshape(meh2,(128,64))
plt.imshow(np.transpose(np.flipud(meh3)), cmap='jet')
plt.colorbar()
plt.title('00000 reshaped')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.grid()
plt.tight_layout()
plt.show()
"""
