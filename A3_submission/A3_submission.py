#LAB 3 barebones

#IMPORTANT: PLEASE READ BEFORE DOING ANYTHING.
#Create your scripts in a way that just requires the TA to run it. 
#ANY script that does not display results at the first attempt,
# and/or that needs input form the TA will suffer a penalization.
#The way the script is expected to work is:
#TA types script name with the image as argument plus parameters and runs on its own


# If you run this code in terminal, use command like python3 Zhao_Vicky_assign3.py.
# If you run this code in colab, please add these two lines, thanks.
#import os
#os.chdir(os.path.dirname(__file__))


import numpy as np
import math
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy import spatial
import skimage
from skimage import io, color, img_as_float, transform, exposure, feature, measure
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from math import floor
from scipy import ndimage


# Part 1: BasicBayer
# We will reconstruct the RGB color image given the corresponding Bayer pattern image.
def part1():
    """add your code here"""
    filename_Grayimage = 'PeppersBayerGray.bmp'
    filename_gridB = 'gridB.bmp'
    filename_gridR = 'gridR.bmp'
    filename_gridG = 'gridG.bmp'    
    # part I
    
    img = io.imread(filename_Grayimage, as_gray =True)
    
    h,w = img.shape
    
    # our final image will be a 3 dimentional image with 3 channels
    rgb = np.zeros((h,w,3),np.uint8);
    
    
    # reconstruction of the green channel IG
    
    IG = np.copy(img) # copy the image into each channel
    
    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            IG[row,col+1]=(int(img[row,col])+int(img[row,col+2]))/2  #B
            IG[row,col+3]=(int(img[row,col+2])+int(img[row+1,col+3]))/2  #D
            IG[row+1,col]=(int(img[row,col])+int(img[row+2,col]))/2  #E
            IG[row+1,col+2]=(int(img[row,col+2])+int(img[row+1,col+1])+int(img[row+1,col+3])+int(img[row+2,col+2]))/4  #G
            IG[row+2,col+1]=(int(img[row+1,col+1])+int(img[row+2,col])+int(img[row+2,col+2])+int(img[row+3,col+1]))/4  #J
            IG[row+2,col+3]=(int(img[row+1,col+3])+int(img[row+3,col+3]))/2  #L
            IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2  #M
            IG[row+3,col+2]=(int(img[row+3,col+1])+int(img[row+3,col+3]))/2  #O
    
    # reconstruction of the red channel IR
    
    IR = np.copy(img) # copy the image into each channel
    
    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            
            IR[row,col+2]=(int(img[row,col+1])+int(img[row,col+3]))/2  #C
            IR[row+1,col+1]=(int(img[row,col+1])+int(img[row+2,col+1]))/2  #F
            IR[row+1,col+2]=(int(img[row,col+1])+int(img[row,col+3])+int(img[row+2,col+1])+int(img[row+2,col+3]))/4  #G
            IR[row+1,col+3]=(int(img[row,col+3])+int(img[row+2,col+3]))/2  #H
            IR[row+2,col+2]=(int(img[row+2,col+1])+int(img[row+2,col+3]))/2  #K
            IR[row,col]=IR[row,col+1]  #A
            IR[row+1,col]=IR[row+1,col+1]  #E
            IR[row+2,col]=IR[row+2,col+1]  #I
            IR[row+3,col+1]=IR[row+2,col+1]  #N
            IR[row+3,col+2]=IR[row+2,col+2]  #O
            IR[row+3,col+3]=IR[row+2,col+3]  #P
            IR[row+3,col]=IR[row+3,col+1]  #M
    
    # reconstruction of the blue channel IB
    
    IB = np.copy(img) # copy the image into each channel
    
    for row in range(0,h,4): # loop step is 4 since our mask size is 4.
        for col in range(0,w,4): # loop step is 4 since our mask size is 4.
            
            IB[row+1,col+1]=(int(img[row+1,col])+int(img[row+1,col+2]))/2  #F
            IB[row+2,col]=(int(img[row+1,col])+int(img[row+3,col]))/2  #I
            IB[row+2,col+1]=(int(img[row+1,col])+int(img[row+1,col+2])+int(img[row+3,col])+int(img[row+3,col+2]))/4  #J
            IB[row+2,col+2]=(int(img[row+1,col+2])+int(img[row+3,col+2]))/2  #K
            IB[row+3,col+1]=(int(img[row+3,col])+int(img[row+3,col+2]))/2  #N
            IB[row,col]=IB[row+1,col]  #A
            IB[row,col+1]=IB[row+1,col+1]  #B
            IB[row,col+2]=IB[row+1,col+2]  #C
            IB[row+1,col+3]=IB[row+1,col+2]  #H
            IB[row+2,col+3]=IB[row+2,col+2]  #L
            IB[row+3,col+3]=IB[row+3,col+2]  #P
            IB[row,col+3]=IB[row,col+2]  #D
    
    # merge the channels
    rgb[:,:,0]=IR
    rgb[:,:,1]=IG
    rgb[:,:,2]=IB
    
    
    plt.imshow(rgb),plt.title('rgb')
    plt.show()    
    
    
# Part 2: Floyd-Steinberg dithering
# Read the RGB image 'lena.png' and implement Floyd-Steinberg dithering to change the representation of this image.
def part2():
    """add your code here"""
    nColours = 8 # The number colours: change to generate a dynamic palette

    imfile = 'lena.png'

    
    image = io.imread(imfile)

    # Strip the alpha channel if it exists
    image = image[:,:,:3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, nColours)
    colours = palette.data
    
    colours = img_as_float([colours.astype(np.ubyte)])[0]
    
    img = FloydSteinbergDitherColor(image, palette)

    plt.imshow(img)
    plt.show()        
    
    
# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]

# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    #print(colours)
    return spatial.KDTree(colours)

# Dynamically calculates and N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, nColours):
    #your code
    #https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    # https://blog.csdn.net/weixin_44911091/article/details/104316204
    # Reshape the image into a pixel list
    height, width, channel = image.shape
    reshape_image = image.reshape(height * width, channel)
    # Get kmeans
    kmeans = KMeans(n_clusters=nColours, random_state=0).fit(reshape_image)
    # Get kmeans center
    colours = kmeans.cluster_centers_
    return makePalette(colours)
  
def FloydSteinbergDitherColor(image, palette):
    
#***** The following pseudo-code is grabbed from Wikipedia: https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.  
#   for each y from top to bottom ==>(height)
#    for each x from left to right ==> (width)
#       oldpixel  := pixel[x][y]
#       newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel
#       pixel[x][y]  := newpixel 
#       quant_error  := oldpixel - newpixel
#       pixel[x + 1][y    ] := pixel[x + 1][y    ] + quant_error * 7 / 16
#       pixel[x - 1][y + 1] := pixel[x - 1][y + 1] + quant_error * 3 / 16
#       pixel[x    ][y + 1] := pixel[x    ][y + 1] + quant_error * 5 / 16
#       pixel[x + 1][y + 1] := pixel[x + 1][y + 1] + quant_error * 1 / 16
    pixel = np.copy(image)
    height, width = pixel.shape[:2]
    for y in range(0, height - 1):
        for x in range(1, width - 1):
            oldpixel = pixel[x][y]
            newpixel = nearest(palette, oldpixel) # Determine the new colour for the current pixel
            pixel[x][y] = newpixel 
            quant_error = oldpixel - newpixel
            pixel[x + 1][y    ] = pixel[x + 1][y    ] + quant_error * 7 / 16
            pixel[x - 1][y + 1] = pixel[x - 1][y + 1] + quant_error * 3 / 16
            pixel[x    ][y + 1] = pixel[x    ][y + 1] + quant_error * 5 / 16
            pixel[x + 1][y + 1] = pixel[x + 1][y + 1] + quant_error * 1 / 16 
    return pixel


# Part 3: perform image rotation and scaling
# Rotation and scaling are specialties of affine transformation,
# and they can be applied to images.    
def part3():
    """add your code here"""
    image1 = io.imread('lab5_img.jpeg')
    # 1. Create a clockwise 90 degrees Rotation transformation matrix T_r.
    # [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]
    T_r = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    # 2. Create a Scale transformation matrix T_s which scales the placement of the points in all directions by two.
    # [[c_x = 2, 0, 0], [0, c_y = 2, 0], [0, 0, 1]]
    T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    
    # 3. Combine the transformations by dot product of T_r and T_s. 
    Combined_T = np.dot(T_r, T_s)
    
    # 4. Apply the combined transformations in one step to the spatial domain of the image data. 
    height1, width1, channel1 = image1.shape
    image2 = np.zeros((2 * height1, 2 * width1, channel1), dtype = np.uint8)
    for x in range(0, height1):
        for y in range(0, width1):
            # get the pixels on (x, y, :)
            origin_data = image1[x, y, :]
            # before_position is a 3 * 1 matrix
            before_position = np.array([[x], [y], [1]])
            # after_position is a 3 * 1 matrix, we get it by dot two matrices
            after_position = np.dot(Combined_T, before_position)
            # get new position (new_x, new_y, :)
            new_x = after_position[0][0]
            new_y = after_position[1][0]
            image2[new_x, new_y, :] = origin_data
    
    # 5. Plot the image after Combined transformation. 
    plt.title('Original lab5_img.jpeg'), plt.imshow(image1,cmap='gray')
    plt.show()
    plt.title('lab5_img.jpeg after Combined transformation'), plt.imshow(image2,cmap='gray')
    plt.show()
    
    # 6.Develop an implementation of nearest neighbour interpolation using inverse Combined_T.
    # Backward mapping
    inverse_Combined_T = np.linalg.inv(Combined_T)
    #image3 = transform.warp(image1, inverse_Combined_T, output_shape=image2.shape, mode='wrap')
    image3 = ndimage.affine_transform(image1, inverse_Combined_T, output_shape=image2.shape, mode ='wrap')
    plt.title('lab5_img.jpeg of nearest neighbour interpolation'), plt.imshow(image3,cmap='gray')
    plt.show()    
    

# Part 4: Use skimage-Python to merge two pictures
# Perform image stitching
def part4():
    """add your code here"""
    image0 = io.imread('im1.jpg', True)
    image1 = io.imread('im2.jpg', True)
    
    plt.imshow(image0,cmap='gray')
    plt.show()
    plt.imshow(image1,cmap='gray')
    plt.show()
    #Feature detection and matching
    
    # Initiate ORB detector
    # your code 
    # https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature.orb#skimage.feature.ORB
    # https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html#sphx-glr-auto-examples-features-detection-plot-orb-py
    detector_extractor = feature.ORB(n_keypoints=3000, fast_threshold=0.05)
    
    # Find the keypoints and descriptors
    # your code #
    detector_extractor.detect_and_extract(image0)
    keypoints0 = detector_extractor.keypoints
    descriptors0 = detector_extractor.descriptors
    
    detector_extractor.detect_and_extract(image1)
    keypoints1 = detector_extractor.keypoints
    descriptors1 = detector_extractor.descriptors
    
    # initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    # your code #
    matches01 = feature.match_descriptors(descriptors0, descriptors1, cross_check=True)
    
    # Compute homography matrix using ransac and ProjectiveTransform
    # your code #
    # model_robust, inliers = ransac ...
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4081273/
    
    from skimage.transform import ProjectiveTransform
    src = keypoints1[matches01[:, 1]][:, ::-1]
    dst = keypoints0[matches01[:, 0]][:, ::-1]
    model_robust, inliers = measure.ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=2)
    
    #Warping
    #Next, we produce the panorama itself. The first step is to find the shape of the output image by considering the extents of all warped images.
    
    r, c = image1.shape[:2]
    
    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])
    
    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)
    
    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))
    
    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    
    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1])
    
    #The images are now warped according to the estimated transformation model.
    
    #A shift is added to ensure that both images are visible in their entirety. Note that warp takes the inverse mapping as input.
    
    from skimage.exposure import rescale_intensity
    from skimage.transform import warp
    from skimage.transform import SimilarityTransform
    
    offset = SimilarityTransform(translation=-corner_min)
    
    image0_ = warp(image0, offset.inverse,
                   output_shape=output_shape)
    
    image1_ = warp(image1, (model_robust + offset).inverse,
                   output_shape=output_shape)
    
    
    #add alpha to the image0 and image1
    
    #your code
    new_image0 = add_alpha(image0_)
    new_image1 = add_alpha(image1_)
    #merge the alpha added image
    
    #your code
    #merged = ...
    merged = new_image0 + new_image1
    
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]
    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.
    
    
    #show and save the output image as '/content/gdrive/My Drive/CMPUT 206 Wi19/Lab5_Files/imgOut.png'
    #your code
    plt.imshow(merged,cmap='gray')
    plt.show()    
    

#An alpha channel is added to the warped images before merging them into a single image:    
def add_alpha(image, background=-1):
    """Add an alpha layer to the image.

    The alpha layer is set to 1 for foreground
    and 0 for background.
    """
    rgb = gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))
    
    

part1()
part2()
part3()
part4()
