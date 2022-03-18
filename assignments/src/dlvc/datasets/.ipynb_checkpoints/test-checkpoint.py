import cv2
from pets import PetsDataset

fdir = 'C:\\Users\\admin\\Desktop\\10. Semester\\Computer Vision\\computer-vision\\assignments\\src\\cifar10'
pds = PetsDataset(fdir, Subset.TRAINING)
print('Xshape, yshape, len, n class: ', pds.X.shape, pds.y.shape, len(pds), pds.num_classes())
print('dtype X,y: ', pds.X.dtype, pds.y.dtype)
print('sample: ', pds[0])
cv2.imshow('sample1', pds.X[0])