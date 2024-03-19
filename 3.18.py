import numpy as np
import cv2
import matplotlib.pyplot as plt

MASK_SIZE = 40
img_src = cv2.imread('man.png',0)
shape = img_src.shape
rows,cols = shape[0],shape[1]
cx,cy = int(shape[0]/2),int(shape[1]/2)
mask= np.zeros((rows,cols, 2),np.uint8)
mask[cx - MASK_SIZE : cx + MASK_SIZE, cy - MASK_SIZE : cy + MASK_SIZE] = 1
img_float = np.float32(img_src)
dft = cv2.dft(img_float,flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift =np.fft.fftshift(dft)
mask_shift =dft_shift * mask
inverse_shift = np.fft.ifftshift(mask_shift)
img_inverse = cv2.idft(inverse_shift)
img_inverse = cv2.magnitude(img_inverse[:,:,0],img_inverse[:,:,1])
plt.subplot(121)
plt.imshow(img_src,cmap='gray')
plt.axis('off')
plt.subplot(122)
plt.imshow(img_inverse, cmap='gray')
plt.axis('off')
plt.savefig('img_inverse.jpg')
plt.show()