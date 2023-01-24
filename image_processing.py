# Python Module to allow for the graffiti images to be processed by the different computer vision algorithms and
# for different numerical values to be extracted from these images based on the features that have been extracted.
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure, segmentation, color
from scipy.stats import circvar
import math


# Creating an Image class, whenever an Image object is created, all of these attribute values are created on
# initialisation
class Image:
    def __init__(self, image_non_resized):
        self.image_non_resized = image_non_resized
        self.image = change_image_size(self.image_non_resized)
        self.image_greyscale = greyscale_image(self.image)
        self.image_segmented = segment_image(self.image)
        self.num_corners = num_corners(self.image)
        self.region_colour_list = segmented_region_colour_list(self.image_segmented)
        self.num_regions = len(self.region_colour_list)
        self.region_colour_variance = segmented_regions_colour_variance(self.region_colour_list)
        self.region_size_list = segmented_regions_size_list(self.image_segmented, self.region_colour_list)
        self.region_size_variance = segmented_regions_size_variance(self.region_size_list)
        self.colourfulness_image = colourfulness(self.image)
        self.image_hog = image_gradients(self.image)
        self.image_canny_edges = canny_edges(self.image)
        self.green, self.blue, self.red = colour_channels(self.image)
        self.average_colour_channels = average_colour_channels(self.green, self.blue, self.red)
        self.average_blue, self.average_green, self.average_red = self.average_colour_channels
        self.green_var, self.blue_var, self.red_var = colour_channel_variance(self.green, self.blue, self.red)
        self.gradient_variance = gradient_circ_variance(self.image_hog)
        self.lines_amount, self.edges_amount = probabilistic_hough_lines(self.image_canny_edges)
        self.straight_line_percent = (self.lines_amount/self.edges_amount)
        self.sift_features = sift_features(self.image)


# Function to change the size of the image to speed up processing and for normalisation of extracted features
def change_image_size(image_original_size):
    dimension_limit = 256
    dimensions_resized = (dimension_limit, dimension_limit)
    image_resized = cv2.resize(image_original_size, dimensions_resized, interpolation=cv2.INTER_AREA)

    return image_resized

# Function to extract SIFT keypoints and SIFT descriptors from images
def sift_features(image):
    image = greyscale_image(image)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # Remove hashtags below to show the SIFT descriptors and keypoints for the image

    #image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return descriptors


# Function to create a greyscale image
def greyscale_image(image_original):
    image_greyscale = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    return image_greyscale


# Function to return the straight edges in an image using the Hough Transform
def hough_lines(image_edges):
    lines = cv2.HoughLines(image_edges, 1, np.pi / 180, 60)
    image_straight_lines = np.zeros((256,256))
    print(lines)
    num_lines = len(lines)

    for i in range(num_lines):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image_straight_lines, (x1, y1), (x2, y2), (255), 1)

    #cv2.imshow('lines', image_straight_lines)
    edge_list = image_edges.flatten()
    edge_list = edge_list.tolist()
    edges = edge_list.count(255)

    straight_list = image_straight_lines.flatten()
    straight_list = straight_list.tolist()
    straight_lines = straight_list.count(255)

    return straight_lines, edges


# Function to return the straight edges in an image using the Probabilistic Hough Transform
def probabilistic_hough_lines(image_edges):
    lines_list= cv2.HoughLinesP(image_edges, 1, np.pi / 180, 50, None, 30, 10)
    image_straight_lines = np.zeros((256,256))

    if lines_list is not None:
        for i in range(0, len(lines_list)):
            l = lines_list[i][0]
            point1 = (l[0], l[1])
            point2 = (l[2], l[3])
            cv2.line(image_straight_lines, point1, point2, (255), 1)
            #cv2.imshow('lines', image_straight_lines)

    edge_list = image_edges.flatten()
    edge_list = edge_list.tolist()
    edges = edge_list.count(255)

    straight_list = image_straight_lines.flatten()
    straight_list = straight_list.tolist()
    straight_lines = straight_list.count(255)

    return straight_lines, edges


# Function to create an image edge map, using the Canny Edge Detector
def canny_edges(image_original):
    threshold_low = 130
    threshold_high = 250
    image_greyscale = greyscale_image(image_original)
    image_blur = cv2.blur(image_greyscale, (2, 2))
    image_edges = cv2.Canny(image_blur, threshold_low, threshold_high)
    #cv2.imshow('edges', image_edges)
    return image_edges


# Function to create the histogram of oriented gradients
def image_gradients(image_original):
    #  section that creates histogram of oriented gradients image
    cell_size = 256
    fd, image_hog = hog(image_original, orientations=36, pixels_per_cell=(cell_size, cell_size),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    max_value = max(image_hog.flatten())
    image_hog_rescaled = exposure.rescale_intensity(image_hog, in_range=(0, 10))
    return image_hog_rescaled


# Function to calculate the Edge Orientation Variance using the histogram of oriented gradients
def gradient_circ_variance(image_hog_rescaled):
    centre = 127
    r = 20
    orientations = 185
    orientation_list = []
    for i in range(0, orientations-1, 1):
        angle = i/orientations
        x = round((centre + r * math.cos(angle * math.pi)))
        y = round((centre + r * math.sin(angle * math.pi)))
        gradient_value = image_hog_rescaled[x,y]
        orientation_list.append(gradient_value)

    orientation_list = list(set(orientation_list))
    orientation_list.remove(0)
    total_gradients = orientation_list + orientation_list

    circular_variance = circvar(total_gradients)
    return circular_variance


# Function to return the Blue, Green and Red colour matrices for an image
def colour_channels(image_original):
    image_dimension = 256
    blue = []
    green = []
    red = []

    for r in range(image_dimension):
        for c in range(image_dimension):
            blue.append(image_original[r, c, 0])
            green.append(image_original[r, c, 1])
            red.append(image_original[r, c, 2])

    return [blue, green, red]


# Function to return the variance of the values of the Blue, Green and Red colour channels for an image
def colour_channel_variance(green, blue, red):
    blue_var = np.var(blue)
    green_var = np.var(green)
    red_var = np.var(red)

    return [blue_var, green_var, red_var]


# Function to return the Blue, Green and Red average colour channel values for an image
def average_colour_channels(blue, green, red):
    image_dimension = 256
    blue = sum(blue) / (image_dimension * image_dimension)
    green = sum(green) / (image_dimension * image_dimension)
    red = sum(red) / (image_dimension * image_dimension)

    return [blue, green, red]


# Function to return number of detected corners in image using the Harris Corner Detector
def num_corners(image_original):
    image_greyscale = greyscale_image(image_original)
    image_greyscale = np.float32(image_greyscale)
    corner_matrix = cv2.cornerHarris(image_greyscale, 2, 3, 0.04)
    threshold = 0.25

    corner_matrix_flat = corner_matrix.flatten()
    corner_list = [i for i in corner_matrix_flat if i >= threshold * corner_matrix.max()]

    num_corners = (len(corner_list))

    return num_corners


# Function to return a segmented image using SLIC Segmentation
def segment_image(image_original):
    image_slic = segmentation.slic(image_original, n_segments=150, start_label=1)
    image_segmented = (color.label2rgb(image_slic, image_original, kind='avg', bg_label=-1)).astype('uint8')

    return image_segmented

# Function to return segmented image using k-means clustering
def segment_image_kmeans(image_original):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 4  # Clusters amount

    attempts = 10

    Z = image_original.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    ret, label, center = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image_original.shape))

    return segmented_image


# Function to create a list of the colours from each of the different regions of a segmented image
def segmented_region_colour_list(image_segmented):
    height, width, channels = image_segmented.shape

    list_segmented_region_colours = []

    for row in image_segmented:
        for pixel in row:  # either row or column
            list_pixels = pixel.tolist()
            if list_pixels not in list_segmented_region_colours:
                list_segmented_region_colours.append(list_pixels)
    return list_segmented_region_colours


#  Function to calculate the variance of the colour of the regions in the segmented image
def segmented_regions_colour_variance(list_segmented_region_colours):
    variance_segmented_regions_colour = np.var(list_segmented_region_colours)
    return variance_segmented_regions_colour


# Function to calculate the size of each segmented region
def segmented_regions_size_list(image_segmented, list_segmented_region_colours): # CHECK THIS FUNCTION WORKS
    height, width = image_segmented.shape[:2]
    flattened_image = image_segmented.reshape(height*width, 3)
    flattened_image = flattened_image.tolist()

    list_size_of_regions=[]
    for region_colour in list_segmented_region_colours:
        amount = flattened_image.count(region_colour)
        list_size_of_regions.append(amount)

    return list_size_of_regions


#  Function to calculate the variance of the sizes of the regions in the segmented image
def segmented_regions_size_variance(list_size_of_regions):
    variance_segmented_regions_size = np.var(list_size_of_regions)
    return variance_segmented_regions_size


#  Function to calculate the colourfulness of an image using
def colourfulness(image):

    # splitting image into the 3 image channels
    (B, G, R) = cv2.split(image.astype('float'))

    # computing rg = R - G
    rg = np.absolute(R - G)
    # computing yb
    yb = np.absolute(0.5 * (R + G) - B)
    # computing the mean and standard deviation
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    std_root = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    mean_root = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # computing the colourfulness
    colourfulness = std_root + (0.3 * mean_root)
    return colourfulness


