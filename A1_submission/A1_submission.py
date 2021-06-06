import os
os.chdir(os.path.dirname(__file__))

from matplotlib import pyplot as plt
import numpy as np
from skimage import exposure, io
import cv2
import math

# Create a histogram by my own way and call Skimage and Numpy histogram functions 
# to compute 256-bin histograms for the same image.
# Plot both histograms side by side to show that they are identical
def part1_histogram_compute():
    """add your code here"""
    # Read the image and get the grayscale image
    image_test = cv2.imread("test.jpg", 0)
    # Get the height and width of the image_test
    height, width = image_test.shape

    # Create My Histogram using 2 for-loop
    H1 = np.zeros(256)
    for i in range(0, height):
      for j in range(0, width):
        H1[image_test[i][j]] += 1

    # Create Skimage Histogram with nbins = 256, source_range='dtype'
    H2, B2 = exposure.histogram(image_test, nbins = 256, source_range='dtype')

    # Create Numpy Histogram
    H3, B3 = np.histogram(image_test, 256, [0, 256])
    
    # Plot 3 histograms
    print("Part I")
    # Create a new figure to display the images
    plt.figure(figsize=(10, 7))
    # Split up the figure into a 1 * 3 grid of sub-figures
    # Display H1 in figure 1
    plt.subplot(131), plt.plot(H1), plt.title('My Histogram')
    plt.xlim([0,256])
    # Display H2 in figure 2
    plt.subplot(132), plt.plot(H2), plt.title('Skimage Histogram')
    plt.xlim([0,256])
    # Display H3 in figure 3
    plt.subplot(133), plt.plot(H3), plt.title('Numpy Histogram')
    plt.xlim([0,256])
    # Show the figure
    plt.show()


# This function is computed to perform histogram equalization on the same test.jpg image used in the last part
# Plot the original image, its histogram, the image after histogram equalization and its histogram
def part2_histogram_equalization():
    """add your code here"""
    # Read the image and get the grayscale image
    image_test = cv2.imread("test.jpg", 0)
    # Get the height and width of the image_test
    height, width = image_test.shape
    
    # Get the histogram of the image by 2 for-loop
    h = np.zeros(256)
    for i in range(0, height):
      for j in range(0, width):
        h[image_test[i][j]] += 1

    # Create a cumulative histogram by the formula
    # H[i] = h[0] for i = 0
    # H[i] = H[i - 1] + h[i] for 0 < i < K
    H = []
    H.append(h[0])
    for i in range(1, 256):
      H.append(H[i - 1] + h[i])

    # Histogram equalization by the formula
    K = 256
    # Create equalImage
    equalImage = np.zeros((height, width))
    for i in range(0, height):
      for j in range(0, width):
        a = image_test[i][j]
        # a' = floor[(K - 1) / MN * H[a] + 0.5]
        equalImage[i][j] = math.floor((K - 1) / (height * width) * H[a] + 0.5)

    # Get the histogram equalHist of the equalization image
    equalHist = np.zeros(256)
    for i in range(0, height):
      for j in range(0, width):
        equalHist[int(equalImage[i][j])] += 1
    
    # Plot 4 pictures
    print("Part II")
    # Create a new figure to display the images
    # Split up the figure into a 2 * 2 grid of sub-figures
    plt.figure(figsize=(10, 8))
    # Display mage_test in figure 1, title 'test.jpg'
    plt.subplot(221), plt.imshow(image_test, 'gray'), plt.title('test.jpg')
    # Display histogram h in figure 2, title 'Original Histogram'
    plt.subplot(222), plt.plot(h), plt.title('Original Histogram')
    # Display equalImage in figure 3, title 'Image after Equalization'
    plt.subplot(223), plt.imshow(equalImage, 'gray'), plt.title('Image after Equalization')
    # Display histogram equalHist in figure 4, title 'Equalization Histogram'
    plt.subplot(224), plt.plot(equalHist), plt.title('Equalization Histogram')
    # Show the figure
    plt.show()


# Compare the histograms of two images day.jpg and night.jpg
# Need to read both images, convert them to grayscale, 
# compute their histograms and print the Bhattacharyya Coefficient of the two histograms
def part3_histogram_comparing():
    """add your code here"""
    # Read the image and get the grayscale image
    day = cv2.imread("day.jpg", 0)
    # Get the height1 and width1 of the day
    height1, width1 = day.shape
    # Read the image and get the grayscale image
    night = cv2.imread("night.jpg", 0)
    # Get the height2 and width2 of the night
    height2, width2 = night.shape

    # Get the histograms of the images
    h1 = np.zeros(256)
    for i in range(0, height1):
      for j in range(0, width1):
        h1[day[i][j]] += 1
    h2 = np.zeros(256)
    for i in range(0, height2):
      for j in range(0, width2):
        h2[night[i][j]] += 1

    # Get M * N
    MN1 = height1 * width1
    MN2 = height2 * width2
    
    # Get Bhattacharyya Coefficient by the formula
    bc = 0
    for i in range(0, 256):
      dayNorm = h1[i] / MN1
      nightNorm = h2[i] / MN2
      bc = bc + math.sqrt(dayNorm * nightNorm)

    # Compute the histograms and print the bhattacharyya coefficient
    print("Part III")
    # Create a new figure to display the images
    # Split up the figure into a 2 * 2 grid of sub-figures
    plt.figure(figsize=(10, 7))
    # Display day in figure 1, title 'day.jpg'
    plt.subplot(221),plt.imshow(day, 'gray'), plt.title('day.jpg')
    # Display histogram h1 in figure 2, title 'Day Histogram'
    plt.subplot(222), plt.plot(h1), plt.title('Day Histogram')
    # Display night in figure 3, title 'night.jpg'
    plt.subplot(223),plt.imshow(night, 'gray'), plt.title('night.jpg')
    # Display histogram h2 in figure 4, title 'Night Histogram'
    plt.subplot(224), plt.plot(h2), plt.title('Night Histogram')
    # Show the figure
    plt.show()
    # Print the Bhattacharyya Coefficient
    print("Bhattacharyya Coefficient is: ", bc)


# Match the histograms of the same two images day.jpg and night.jpg
# First Grayscale version, then RGB version    
def part4_histogram_matching():
    # Read the images and get the grayscale images
    day = cv2.imread("day.jpg", 0)
    # Get the height1 and width1 of the day
    height1, width1 = day.shape
    night = cv2.imread("night.jpg", 0)
    # Get the height2 and width2 of the night
    height2, width2 = night.shape
    
    # Get the histogram of the images
    '''
    h1 = np.zeros(256)
    for i in range(0, height1):
      for j in range(0, width1):
        h1[day[i][j]] += 1
    h2 = np.zeros(256)
    for i in range(0, height2):
      for j in range(0, width2):
        h2[night[i][j]] += 1
    '''
    # We can also use build-in function to do this
    h1, B1 = np.histogram(day, 256, [0, 256])   # Use build-in histogram function
    h2, B2 = np.histogram(night, 256, [0, 256])

    # Create darkerDay image
    darkerDay = np.zeros_like(day)

    # Create a cumulative histogram of the day image and the night image
    '''
    cdf_day = []
    cdf_day.append(h1[0])
    for i in range(1, 256):
      cdf_day.append(cdf_day[i - 1] + h1[i])

    cdf_night = []
    cdf_night.append(h2[0])
    for i in range(1, 256):
      cdf_night.append(cdf_night[i - 1] + h2[i])
    '''
    # We can also use build-in function to do this
    cdf_day = np.cumsum(h1)    # Use build-in histogram function
    cdf_night = np.cumsum(h2)

    # Matching the picture by the formula
    for i in range(0, height1):
      for j in range(0, width1):
        a = day[i][j]
        aa = 0
        while cdf_day[a] > cdf_night[aa]:
          aa += 1
        darkerDay[i][j] = aa
        
    # Get the histogram of the darkerDay image
    h3 = np.zeros(256)
    for i in range(0, height1):
      for j in range(0, width1):
        h3[darkerDay[i][j]] += 1
 
    # Compute the histograms
    print("Part IV (a)")
    # Create a new figure to display the images
    # Split up the figure into a 3 * 2 grid of sub-figures
    plt.figure(figsize=(20, 14))
    # Display day in figure 1, title 'day.jpg'
    plt.subplot(321),plt.imshow(day, 'gray'), plt.title('day.jpg')
    # Display histogram h1 in figure 2, title 'Day Histogram'
    plt.subplot(322), plt.plot(h1), plt.title('Day Histogram')
    # Display night in figure 3, title 'night.jpg'
    plt.subplot(323),plt.imshow(night, 'gray'), plt.title('night.jpg')
    # Display histogram h2 in figure 4, title 'Night Histogram'
    plt.subplot(324), plt.plot(h2), plt.title('Night Histogram')
    # Display darkerDay in figure 5, title 'Darker version of the day.jpg'
    plt.subplot(325),plt.imshow(darkerDay, 'gray'), plt.title('Darker version of the day.jpg')
    # Display histogram h3 in figure 6, title 'Matched Day Histogram'
    plt.subplot(326), plt.plot(h3), plt.title('Matched Day Histogram')
    # Show the figure
    plt.show()


    # Read the images and get the grayscale images
    day = io.imread("day.jpg")
    # Get the height1 and width1 of the day
    height1, width1, channel1 = day.shape
    night = io.imread("night.jpg")
    # Get the height2 and width2 of the night
    height2, width2, channel2 = night.shape

    # Create darkerDay image
    darkerDay = np.zeros_like(day)

    # Print RGB matched images
    print("Part IV (b)")

    # Channel: 0
    print('Channel: ', 0)
    # Use build-in functions to get the histograms and cumulative histograms
    h1, B1 = np.histogram(day[:, :, 0], 256, [0, 256]) 
    h2, B2 = np.histogram(night[:, :, 0], 256, [0, 256])
    cdf_day = np.cumsum(h1)  
    cdf_night = np.cumsum(h2)
    # Matching the channel 0 of day[:, :, 0] and night[:, :, 0]
    for i in range(0, height1):
      for j in range(0, width1):
        a = day[i][j][0]
        aa = 0
        while cdf_day[a] > cdf_night[aa]:
          aa += 1
        darkerDay[i][j][0] = aa

    # Create a new figure to display the images
    # Split up the figure into a 1 * 3 grid of sub-figures
    plt.figure(figsize=(20, 14))
    # Display day[:, :, 0] in figure 1, title 'day.jpg'
    plt.subplot(131),plt.imshow(day[:, :, 0], 'gray'), plt.title('day.jpg')
    # Display night[:, :, 0] in figure 2, title 'night.jpg'
    plt.subplot(132),plt.imshow(night[:, :, 0], 'gray'), plt.title('night.jpg')
    # Display darkerDay[:, :, 0] in figure 3, title 'Darker version of the day.jpg'
    plt.subplot(133),plt.imshow(darkerDay[:, :, 0], 'gray'), plt.title('Darker version of the day.jpg')
    # Show the figure
    plt.show()

    # Channel: 1
    print('Channel: ', 1)
    # Use build-in functions to get the histograms and cumulative histograms
    h1, B1 = np.histogram(day[:, :, 1], 256, [0, 256]) 
    h2, B2 = np.histogram(night[:, :, 1], 256, [0, 256])
    cdf_day = np.cumsum(h1)  
    cdf_night = np.cumsum(h2)
    # Matching the channel 1 of day[:, :, 1] and night[:, :, 1]
    for i in range(0, height1):
      for j in range(0, width1):
        a = day[i][j][1]
        aa = 0
        while cdf_day[a] > cdf_night[aa]:
          aa += 1
        darkerDay[i][j][1] = aa

    # Create a new figure to display the images
    # Split up the figure into a 1 * 3 grid of sub-figures
    plt.figure(figsize=(20, 14))
    # Display day[:, :, 1] in figure 1, title 'day.jpg'
    plt.subplot(131),plt.imshow(day[:, :, 1], 'gray'), plt.title('day.jpg')
    # Display night[:, :, 1] in figure 2, title 'night.jpg'
    plt.subplot(132),plt.imshow(night[:, :, 1], 'gray'), plt.title('night.jpg')
    # Display darkerDay[:, :, 1] in figure 3, title 'Darker version of the day.jpg'
    plt.subplot(133),plt.imshow(darkerDay[:, :, 1], 'gray'), plt.title('Darker version of the day.jpg')
    # Show the figure
    plt.show()

    # Channel: 2
    print('Channel: ', 2)
    # Use build-in functions to get the histograms and cumulative histograms
    h1, B1 = np.histogram(day[:, :, 2], 256, [0, 256]) 
    h2, B2 = np.histogram(night[:, :, 2], 256, [0, 256])
    cdf_day = np.cumsum(h1)  
    cdf_night = np.cumsum(h2)
    # Matching the channel 2 of day[:, :, 2] and night[:, :, 2]
    for i in range(0, height1):
      for j in range(0, width1):
        a = day[i][j][2]
        aa = 0
        while cdf_day[a] > cdf_night[aa]:
          aa += 1
        darkerDay[i][j][2] = aa

    # Create a new figure to display the images
    # Split up the figure into a 1 * 3 grid of sub-figures
    plt.figure(figsize=(20, 14))
    # Display day[:, :, 2] in figure 1, title 'day.jpg'
    plt.subplot(131),plt.imshow(day[:, :, 2], 'gray'), plt.title('day.jpg')
    # Display night[:, :, 2] in figure 2, title 'night.jpg'
    plt.subplot(132),plt.imshow(night[:, :, 2], 'gray'), plt.title('night.jpg')
    # Display darkerDay[:, :, 2] in figure 3, title 'Darker version of the day.jpg'
    plt.subplot(133),plt.imshow(darkerDay[:, :, 2], 'gray'), plt.title('Darker version of the day.jpg')
    # Show the figure
    plt.show()

    # Put together the resultant matched channels into an RGB image
    print('Finally')
    # Create a new figure to display the images
    # Split up the figure into a 1 * 3 grid of sub-figures
    plt.figure(figsize=(20, 14))
    # Display day in figure 1, title 'ource_rgb'
    plt.subplot(131),plt.imshow(day), plt.title('source_rgb')
    # Display night in figure 2, title 'template_rgb'
    plt.subplot(132),plt.imshow(night), plt.title('template_rgb')
    # Display darkerDay in figure 3, title 'matched_rgb'
    plt.subplot(133),plt.imshow(darkerDay), plt.title('matched_rgb')
    # Show the figure
    plt.show()    



if __name__ == '__main__':
    part1_histogram_compute()
    part2_histogram_equalization()
    part3_histogram_comparing()
    part4_histogram_matching()
