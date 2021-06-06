#LAB 2 barebones

#IMPORTANT: PLEASE READ BEFORE DOING ANYTHING.
#Create your scripts in a way that just requires the TA to run it. 
#ANY script that does not display results at the first attempt,
# and/or that needs input form the TA will suffer a penalization.
#The way the script is expected to work is:
#TA types script name with the image as argument plus parameters and runs on its own


# If you run this code in terminal, use command like python3 Zhao_Vicky_a2.py.
# If you run this code in colab, please add these two lines, thanks.
#import os
#os.chdir(os.path.dirname(__file__))

from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import skimage
from skimage import feature
import cv2


# Filter the grayscale image with the given filters by using scikit-image/python
# Create a 3-by-3 matrix and write code to implement these filters
# We need to do the Convolution by ourselves instead of using build-in function
# https://www.pythonheidong.com/blog/article/772811/6efcf94a501d9b8b7e7e/
def part1():
    """add your code here"""
    # First, read the image and get the grayscale image.
    # Get the height and width of the moon.
    moon = cv2.imread('moon.png', 0)
    moon_height, moon_width = moon.shape
    
    # Step 1: Initial laplacian filter matrix,
    # get the height and width of the filter, get the values of h and w,
    # and get the values of H and W.
    # Let H and W be the shape of padded image I, 
    # and we need to copy the image to the center of I, we use slice way to do it.
    K_laplacianFilter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    K_height, K_width = K_laplacianFilter.shape
    h = int((K_height - 1) / 2)
    w = int((K_width - 1) / 2)
    H = moon_height + 2 * h
    W = moon_width + 2 * w
    # Get the padded image I, we can also use np.pad() here.
    #I = np.pad(moon, pad_width = 1, mode = 'symmetric')
    I = np.zeros((H, W))
    I[h:(H-h), w:(W-w)] = moon

    # Step 2: Initialize output image J to all zeros.
    # Matrix J has shape (H, W), so we make it like I, zero matrix.
    J = np.zeros_like(I)

    # Step 3: Convolution by four for-loops instead of using build-in function
    for i in range(h, H - h):
      for j in range(w, W - w):
        for m in range(-h, h + 1):
          for n in range(-w, w + 1):
            J[i, j] += I[i + m, j + n] * K_laplacianFilter[m + h, n + w]

    # Step 4: Strip padding from J, using slice way again
    # so that J has shape (H - 2h, W - 2w), return J
    strip_padding_J = np.zeros_like(moon)
    strip_padding_J = J[h:(H-h), w:(W-w)]

    # Repeat the above steps
    # Change matrix into [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    K_2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    K_height, K_width = K_2.shape
    h = int((K_height - 1) / 2)
    w = int((K_width - 1) / 2)
    H = moon_height + 2 * h
    W = moon_width + 2 * w
    I = np.zeros((H, W))
    I[h:(H-h), w:(W-w)] = moon
    
    J = np.zeros_like(I)
    
    for i in range(h, H - h):
      for j in range(w, W - w):
        for m in range(-h, h + 1):
          for n in range(-w, w + 1):
            J[i, j] += I[i + m, j + n] * K_2[m + h, n + w]
    
    strip_padding_J2 = np.zeros_like(moon)
    strip_padding_J2 = J[h:(H-h), w:(W-w)]

    # Repeat the above steps
    # Change matrix into [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    K_3 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    K_height, K_width = K_3.shape
    h = int((K_height - 1) / 2)
    w = int((K_width - 1) / 2)
    H = moon_height + 2 * h
    W = moon_width + 2 * w
    I = np.zeros((H, W))
    I[h:(H-h), w:(W-w)] = moon
    
    J = np.zeros_like(I)
    
    for i in range(h, H - h):
      for j in range(w, W - w):
        for m in range(-h, h + 1):
          for n in range(-w, w + 1):
            J[i, j] += I[i + m, j + n] * K_3[m + h, n + w]
    
    strip_padding_J3 = np.zeros_like(moon)
    strip_padding_J3 = J[h:(H-h), w:(W-w)]

    # Mean filter, so the matrix is equal to 
    # [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    # Following the formula Im + (Im - Im * average_filter)
    # The differences of this filter with the above three is 
    # we need to copy the image moon to matrix J first instead of making a zero matrix J,
    # and we also need to add image moon to J again after we finish convolution step.
    K_4 = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    K_height, K_width = K_4.shape
    h = int((K_height - 1) / 2)
    w = int((K_width - 1) / 2)
    H = moon_height + 2 * h
    W = moon_width + 2 * w
    I = np.zeros((H, W))
    I[h:(H-h), w:(W-w)] = moon
    
    J = np.zeros_like(I)
    J[h:(H-h), w:(W-w)] = moon
   
    for i in range(h, H - h):
      for j in range(w, W - w):
        for m in range(-h, h + 1):
          for n in range(-w, w + 1):
            J[i, j] = J[i, j] - I[i + m, j + n] * K_4[m + h, n + w]  
    J[h:(H-h), w:(W-w)] += moon       
    
    strip_padding_J4 = np.zeros_like(moon)
    strip_padding_J4 = J[h:(H-h), w:(W-w)]

    plt.figure(figsize=(20, 15))
    # Split up the figure into a 3 * 2 grid of sub-figures
    # Display original moon.png in figure 1
    plt.subplot(321), plt.imshow(moon, 'gray'), plt.title('moon.png')
    # Display laplacianFilter moon.png in figure 3
    plt.subplot(323), plt.imshow(strip_padding_J, 'gray'), plt.title('moon.png after laplacianFilter')
    # Display Filter2 moon.png in figure 4
    plt.subplot(324), plt.imshow(strip_padding_J2, 'gray'), plt.title('moon.png after Filter2')
    # Display Filter3 moon.png in figure 5
    plt.subplot(325), plt.imshow(strip_padding_J3, 'gray'), plt.title('moon.png after Filter3')
    # Display Filter4 moon.png in figure 6
    plt.subplot(326), plt.imshow(strip_padding_J4, 'gray'), plt.title('moon.png after Filter4')
    # Show the figure
    plt.show()
    

# Read noisy.jpg corrupted with salt and pepper noise by io.imread() function, 
# and apply a median filter and a Gaussian filter to remove the noise,
# compare and see the diffferences between two filters.
# Based on the output images, I think median filter is more successful.
# https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median
# https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
def part2():
    """add your code here"""
    noisy = io.imread('noisy.jpg')
    median = skimage.filters.median(noisy)
    Gaussian = skimage.filters.gaussian(noisy, multichannel=True)
    # From the outputs, we can find that Median filter is more successful

    plt.figure(figsize=(20, 14))
    plt.subplot(131), plt.imshow(noisy, 'gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(median, 'gray'), plt.title('median filter')
    plt.subplot(133), plt.imshow(Gaussian, 'gray'), plt.title('Gaussian filter')
    plt.show()


# Simple image inpainting: we need to repair the broken image.
# Set J = I first, and then repeat (I set the iteration equals to 100).
# In every iteration, use skimage.filters.gaussian() to smooth damaged J,
# and then copy good pixels from I to J, and go to next iteration.
def part3():
    """add your code here"""
    I = io.imread('damage_cameraman.png')
    I_height, I_width = I.shape[:2]
    U = io.imread('damage_mask.png')
    U_height, U_width = U.shape[:2]
    J = io.imread('damage_cameraman.png')
    
    for i in range(0, 100):
      # J = GaussianSmooth(J) # Smooth damaged J
      J = skimage.filters.gaussian(J, 3, multichannel=True)
      for k in range(0, I_height):
        for j in range(0, I_width):
          if U[k, j] != 0:
            # J(U) = I(U) # Copy good pixels
            J[k, j] = I[k, j]

    plt.figure(figsize=(20, 14))
    plt.subplot(131), plt.imshow(I, 'gray'), plt.title('damage_cameraman.png')
    plt.subplot(132), plt.imshow(U, 'gray'), plt.title('damage_mask.png')
    plt.subplot(133), plt.imshow(J, 'gray'), plt.title('Output')
    plt.show()


# Edges
# Read the gray scale image ex2.jpg by io.imread() and display the image ex2.jpg,
# compute gradient of the image (both the horizontal derivative and vertical derivative) by Sobel operators,
# and display the horizontal and vertical direction derivative images,
# and compute gradient magnitude image by suitably combining the horizontal and the vertical derivative images
# display the gradient magnitude image.
def part4():
    """add your code here"""
    ex2 = io.imread('ex2.jpg', as_gray = True)   
    ex2_height, ex2_width = ex2.shape

    # Display the horizontal and vertical direction derivative images
    # Compute gradient magnitude image by suitably combining the horizontal and the vertical derivative images
    ex2_v = skimage.filters.sobel_v(ex2)
    ex2_h = skimage.filters.sobel_h(ex2)

    # Display the gradient magnitude image by the formula
    gradientMagnitude = (ex2_h ** 2 + ex2_v ** 2) ** (1/2)

    plt.figure(figsize=(20, 14))
    plt.subplot(221), plt.imshow(ex2, 'gray'), plt.title('ex2.jpg')
    plt.subplot(222), plt.imshow(ex2_v, 'gray'), plt.title('vertical derivative image')
    plt.subplot(223), plt.imshow(ex2_h, 'gray'), plt.title('horizontal derivative image')
    plt.subplot(224), plt.imshow(gradientMagnitude, 'gray'), plt.title('gradient magnitude image')
    plt.show()
    

# Canny edge detector
# Read the gray scale image ex2.jpg by io.imread() function,
# display the image ex2_G which is after gaussian smooth, here I choose skimage.filters.gaussian(ex2, 1, multichannel=True),
# then do gradient to ex2_G like part 4, 
# study the provided code and understand the effect of threshold values and the effect of sigma value,
# I use skimage.feature.canny() here to show the effect of threshold values and sigma value.
# Before we use skimage.feature.canny(), we need to convert an image to 8-bit unsigned integer format, which is in the range of 0 to 255.
def part5():
    """add your code here"""
    ex2 = io.imread('ex2.jpg', as_gray = True)
    ex2_height, ex2_width = ex2.shape

    # Smooth
    ex2_G = skimage.filters.gaussian(ex2, 1, multichannel=True)
    plt.figure(figsize=(20, 14))
    plt.subplot(121), plt.imshow(ex2, 'gray'), plt.title('original ex2.jpg')
    plt.subplot(122), plt.imshow(ex2_G, 'gray'), plt.title('smoothed image of ex2.jpg')
    plt.show()

    # Gradient
    ex2_v = skimage.filters.sobel_v(ex2_G)
    ex2_h = skimage.filters.sobel_h(ex2_G)
    gradientMagnitude = (ex2_h ** 2 + ex2_v ** 2) ** (1/2)

    # Transfer the image pixel data into uint8 type
    gradientMagnitude = skimage.img_as_ubyte(gradientMagnitude)

    # To understand the effect of threshold values
    # fix sigma = 1.0, use matplotlib.pyplot.subplot to plot the following 4 figures together and see their difference : 
    # low threshold = 25; low threshold = 50; high threshold = 150; high threshold = 200
    ex2_1 = skimage.feature.canny(gradientMagnitude, sigma=1, low_threshold=25)
    ex2_2 = skimage.feature.canny(gradientMagnitude, sigma=1, low_threshold=50)
    ex2_3 = skimage.feature.canny(gradientMagnitude, sigma=1, high_threshold=150)
    ex2_4 = skimage.feature.canny(gradientMagnitude, sigma=1, high_threshold=200)
    plt.figure(figsize=(20, 14))
    plt.subplot(221), plt.imshow(ex2_1, 'gray'), plt.title('low_threshold=25')
    plt.subplot(222), plt.imshow(ex2_2, 'gray'), plt.title('low_threshold=50')
    plt.subplot(223), plt.imshow(ex2_3, 'gray'), plt.title('high_threshold=150')
    plt.subplot(224), plt.imshow(ex2_4, 'gray'), plt.title('high_threshold=200')
    plt.show()

    # To understand the effect of sigma value
    # fix low_threshold=50 and high_threshold=150, use matplotlib.pyplot.subplot to plot the following 4 figures together and see their difference :
    # sigma =  1.0; sigma = 1.5; sigma = 2.0; sigma = 2.5
    ex2_5 = skimage.feature.canny(gradientMagnitude, sigma=1.0, low_threshold=50, high_threshold=150)
    ex2_6 = skimage.feature.canny(gradientMagnitude, sigma=1.5, low_threshold=50, high_threshold=150)
    ex2_7 = skimage.feature.canny(gradientMagnitude, sigma=2.0, low_threshold=50, high_threshold=150)
    ex2_8 = skimage.feature.canny(gradientMagnitude, sigma=2.5, low_threshold=50, high_threshold=150)
    plt.figure(figsize=(20, 14))
    plt.subplot(221), plt.imshow(ex2_5, 'gray'), plt.title('sigma=1.0')
    plt.subplot(222), plt.imshow(ex2_6, 'gray'), plt.title('sigma=1.5')
    plt.subplot(223), plt.imshow(ex2_7, 'gray'), plt.title('sigma=2.0')
    plt.subplot(224), plt.imshow(ex2_8, 'gray'), plt.title('sigma=2.5')
    plt.show()

    # The conclusion is:
    # Fix sigma = 1.0 and we change the threshold value, 
    # the larger the threshold value, the fewer edges appear, 
    # the edges disappear evenly on the image,
    # secondary edges will disappear first, most main edges can be kept.

    # Fix low_threshold=50 and high_threshold=150 and we change the sigma value, 
    # the larger the sigma value, the worse the integrity of the boundary,
    # the edges disappears fast if we make the sigma value larger,
    # the edges don't disappear evenly on the image,
    # we cannot tell what the original object is.

    
    
    
part1()
part2()
part3()
part4()
part5()
