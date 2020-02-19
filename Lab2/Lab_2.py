import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy import ndimage
import math

np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
plt.close('all')# close previously opened pictures

filein = 'melanoma_27.jpg'
im = mpimg.imread(filein)   # Import the image from the jpg file
# Plot the original image
plt.figure(1)
plt.imshow(im)
plt.title('Original Image')
plt.show()

#%% reshape the image from 3D to 2D

N1, N2, N3 = im.shape   # Define the length of the 3 Dimensions of the image
im_2D = im.reshape((N1*N2, N3))  # N1*N2 rows and N3 columns

#%% get a simplified image with only Ncluster colors

Ncluster = 3
kmeans = KMeans(n_clusters=Ncluster, random_state=0)
kmeans.fit(im_2D)   # Apply the hard K means
centroids = kmeans.cluster_centers_.astype('uint8') # The 3 principal colors found in the image

# Quantization of the image
im_2D_quant=im_2D.copy()    # Create a copy of im_2D
for kc in range(Ncluster):
    quant_color_kc = centroids[kc, :]   # The values for the kc color
    ind = np.argwhere(kmeans.labels_ == kc)    # The indexes of im_2D that belong to class kc
    im_2D_quant[ind, :] = quant_color_kc    # The color of the elements belonging to class kc is changed
im_quant = im_2D_quant.reshape((N1, N2, N3))
plt.figure(2)
plt.imshow(im_quant, interpolation=None)
plt.title('image with quantized colors')
plt.show()  # Plot with the quantized image

#%% Preliminary steps to find the contour after the clustering

# Now I have to distinguish the real mole by using the hard KMeans again
ind_dark_color = np.argmin(centroids.sum(axis=1))   # The index of the darkest centroid color
matr_ind = kmeans.labels_.reshape(N1, N2)   # in position i, j there is the cluster which the pixel belongs to
ind_dark = np.argwhere(matr_ind == ind_dark_color)   # Pixels belonging to the cluster of the darkest color

# ask the user to write the number of objects belonging to
# cluster ind_dark_color in the image with quantized colors
N_spots = input('The number of darkest spot you see in the original image: ')
N_spots = int(N_spots)

# Find the centre of the mole
if N_spots == 1:
    center_mole = np.median(ind_dark, axis=0).astype(int)
else:
    # use K-means to get the N_spots clusters of ind_dark
    kmeans_2 = KMeans(n_clusters=N_spots, random_state=0)
    kmeans_2.fit(ind_dark)
    centroids2 = kmeans_2.cluster_centers_.astype(int)
    # the mole is in the middle of the picture:
    center_image = np.array([N1/2, N2/2])
    center_image.shape = (1, 2)
    dist = np.zeros((N_spots, 1), dtype=float)
    for j in range(N_spots):
        dist[j] = np.linalg.norm(center_image-centroids2[j, :])
    center_mole = centroids2[dist.argmin(), :]

# Take a subset of the image that include the mole
cond = True
c0 = center_mole[0]
c1 = center_mole[1]
RR, CC = matr_ind.shape
stepmax = min([c0,RR-c0,c1,CC-c1])
matr_sel = (matr_ind == ind_dark_color) # matr_sel is a boolean NDarray with N1 rows and N2 columns
matr_sel = matr_sel*1 # im_sel is now an integer NDarray with N1 rows and N2 columns
area0 = 0
surf0 = 1
step = 10 # each time the algorithm increases the area by 2*step pixels
# horizontally and vertically
while cond:
    subset = matr_sel[c0 - step:c0 + step + 1, c1 - step:c1 + step + 1]
    area = np.sum(subset)
    Delta = np.size(subset) - surf0
    surf0 = np.size(subset)
    if area > area0 + 0.01*Delta:
        step = step + 10
        area0 = area
        cond = True
        if step > stepmax:
            cond = False
    else:
        cond = False
# subset is the search area
plt.matshow(subset, cmap='winter')
plt.title("Cropped Quantized Image")
plt.show()

#%% Preliminary step to find the borders of the mole
# subset = ndimage.morphology.binary_fill_holes(subset).astype(int)

#%% Cleaning of the image obtained
# A submatrix is scrolled over the image.
i0 = j0 = 40 # The maximum size of the submatrix
# This submatrix removes the spots situated out of the surface
for k in range(i0-1, 1, -1):
    for i in range(subset.shape[0]-i0+k):
        for j in range(subset.shape[1]-j0): # The initial size of the submatrix is 2x2, then it increases up to the threshold
            sub = subset[i: i+i0-k, j:j+j0-k]
            sub = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
            if 1 not in sub:
                subset[i:i+i0-k, j:j+j0-k] = 0

# This submatrix removes the holes inside the mole.
for k in range(i0-1, 1, -1):
    for i in range(subset.shape[0]-i0+k):
        for j in range(subset.shape[1]-j0):
            sub = subset[i: i+i0-k, j:j+j0-k]
            sub = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
            if 0 not in sub:
                subset[i:i+i0-k, j:j+j0-k] = 1

plt.matshow(subset, cmap='winter')
plt.title("Cleaned Quantized Image")
plt.show()
#%% Find the borders

borders = np.zeros((subset.shape[0], subset.shape[1]), dtype=int)
# The borders are detected through the differences between adjacent pixels.
for i in range(subset.shape[0]):  # Difference between adjacent pixels row by row
    for j in range(subset.shape[1]-1):
        borders[i, j] = abs(subset[i, j+1] - subset[i, j])

for j in range(subset.shape[1]):  # Difference between adjacent pixels column by column
    for i in range(subset.shape[0]-1):
        borders[i, j] += abs(subset[i+1, j] - subset[i, j])

for i in range(subset.shape[0]):  # Allows to have only pixels equal to 1 or 0
    for j in range(subset.shape[1]-1):
        if borders[i, j] != 0:
            borders[i, j] = 1

plt.matshow(borders, cmap='winter')
plt.title("Perimeter")
plt.show()

#  Total area of the mole, obtained from the Cleaned Quantized Image
tot_area = np.argwhere(subset == 1).shape[0]
#  The perimeter of the mole obtained from the Perimeter image
perimeter = np.argwhere(borders == 1).shape[0]
print(perimeter, tot_area)
# The radius of a circle with the same area of the mole.
r_eq = math.sqrt(tot_area/math.pi)
# The ratio between the perimeter of the mole and the perimeter of the circle with the same area
ratio = (perimeter)/(2*math.pi*r_eq)
print(round(ratio, 3))





