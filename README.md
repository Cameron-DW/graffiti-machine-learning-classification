
# Graffiti Image Classifier

This is an Image Classification Tool that can be used to classify images of graffiti into either Artwork or Vandalism. It was created using Python, OpenCV, TensorFlow, sci-kit-learn and MATLAB.

Artwork Graffiti           |  Vandalism Graffiti
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/89137794/214317482-f42b75a3-2b41-485e-83f8-d14f2caaf551.png" width= "300"> | <img src="https://user-images.githubusercontent.com/89137794/214317529-65c20eaa-0508-4d6d-90db-662cbcb49f01.png" width= "300"> 

## Motivation

The aim of this work was to create a method that used computer vision alongside machine learning to allow for images of graffiti to be classified based on if they were artworks or forms of vandalism. Graffiti can be considered art as much as any other artistic medium. However, even so, all types of graffiti are currently considered a criminal offence in England under the Criminal Damage Act 1971. The government aims to remove both artwork graffiti and vandalism graffiti through graffiti clean-up. This clean-up is very expensive; costing £1 billion in 2015 for the UK alone. However, if only vandalism graffiti was removed then the graffiti clean-up cost would be reduced. This approach is starting to be considered; world-famous artist Banksy’s graffiti was removed by the Tendering District council who later stated, after realising the graffiti was created by Banksy, that they would be delighted if he returned to create more which wouldn’t be removed. Therefore the motivations behind this work were to create a classifier that would classify graffiti images and therefore allow for governmental bodies to determine if specific graffiti is considered vandalism and should be removed. This could reduce the cost of graffiti clean-up which would allow for governments to allocate some of the current graffiti clean-up budget elsewhere.

## Differerent Computer Vision Algorithms Used
398 labelled images were gathered, 199 of Artwork Graffiti and 199 of Vandalism Graffiti. Different computer vision algorithms were performed on all of these and from these algorithms, numerical values were calculated and used as input to train a neural network. Different combinations of the computer visions algorithm's numerical values were used to train the neural network to find the best inputs for the highest accuracy classifications. Different parameter values were tested for each of the computer vision algorithms that were used to extract the image features to ensure that the correct numerical feature values were being computed.

### SIFT Features
SIFT keypoints and SIFT descriptors were extracted from all of the labelled imagesn. These SIFT descriptors were used to make a basic classification system where the SIFT descriptors were matched between different images within the same class to determine if there were any similarities. If there were similarities between the extracted SIFT descriptors between images in the same class then a Bag-of-Features / Bag-of-Words (BOW) approach with SIFT and an SVM (Support Vector Machine) would give positive results for a graffiti image classifier. However due to poor matching between SIFT features between images within the same class, implementing this Bag-of-Features SIFT approach with an SVM was not used.

Original Image       |  Detected Corners
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/89137794/214323984-1385433d-ed6a-4dc1-bf7a-801830dbb221.png" width="250"> | <img src="https://user-images.githubusercontent.com/89137794/214288257-69bf605a-e059-4cd1-9b00-c8187c8bec9b.png" width="250">

### Image Segmentation
K-means image segmentation was initially used as the unsupervised image segmentation algorithm, with different k values being tested to determine how many clusters should be created for the images to be segmented into. Although k-means segmentation did segment the image, it did so by comparing different pixel BGR colour values, not including the pixel’s x and y location within the image. Therefore it just created many disconnected segmented regions throughout the image of k different colour values. Therefore k-means segmentation was not effective for this classifier as the regions needed to be based on both colour and location in order for specific desired image features values to be extracted.

SLIC segmentation was implemented which used the 5D space of colour by including 3 colour channels (BGR) and the x and y location of a pixel, the segmentation through clustering was applied using all of these dimensions. Therefore SLIC segmentation was more effective at comparing regions within an image rather than just segmenting images into different colours which resulted in regions of the same colour being very spatially distant.

Original Image         |  K-Means Segmentation | SLIC Segmentation
:-------------------------:|:-------------------------:|-------------
<img src="https://user-images.githubusercontent.com/89137794/214317723-d26b70bb-29e0-4fb5-877e-880d9bc2bc5b.png" width="250">| <img src="https://user-images.githubusercontent.com/89137794/214317736-241ea89f-3f39-46fb-9154-e2cc53965008.png" width = "250"> | <img src="https://user-images.githubusercontent.com/89137794/214317753-358168fe-14a3-4ac9-bc3c-a892b23b013c.png" width = "250">

### Harris Corner Detector
Different parameters for the Harris Corner detector were used on images with a known number of corners. Visual representations for the locations of detected corners within images were used during evaluation so that false negatives and false positives for the corner detector could be easily identified. Many false positive corner detections were located in areas of the image with a lot of noise such as due to a strong light source on a textured background, and therefore the Gaussian Blur was increased in order to stop corners incorrectly being identified in noisy image sections. The accuracy of the implemented Harris corner detector was high and it required very minimal computation time.

Original Image       |  Detected Corners
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/89137794/214323984-1385433d-ed6a-4dc1-bf7a-801830dbb221.png" width ="250"> |  <img src="https://user-images.githubusercontent.com/89137794/214324004-3e6a547e-abd9-4eed-b2b9-83252acba400.png" width="250">

### Histogram of Oriented Gradients (Variance in Edge Orientations)
The variance of the different edge orientations within the images were calculated through implementing the histogram of oriented gradients algorithm using OpenCV. 36 orientations were used, so there was an edge orientation at every 5 degrees. This algorithm would go through the image and identify the quantity of edges within the image at each of these 36 orientations. It returned a list for the quantity of edges that were angled in each orientation in the correct order from angle 0 to 𝜋. These edge orientation quantities can be viewed as a histogram/graph.

<img src="https://user-images.githubusercontent.com/89137794/214328607-14c53f05-ddfc-44d0-8049-4c3ece10732c.png" width="450">

and the variance of this can be calculated.

### Canny Edge Detector
Different parameters were used for the Canny edge detector such as different thresholds and gaussian blurs, each was evaluated to create a detector that reduced the number of incorrectly detected edges and the number of disconnected edges. Using a Canny edge detector with only a single threshold value gave poor edge detection results. Using a hysteresis threshold for the Canny edge detector produced better results. It caused fewer disconnected edges to form and fewer edges being incorrectly detected from areas with noise. Implementing a Gaussian blur of kernel size (2,2) on the image before it was used with the Canny Edge Detection algorithm produced the best results. It did not blur the image too much so that edges could no longer be detected but it did blur the image enough so that noise in the image was reduced therefore not being detected as an edge.

|Single Threshold         |  Hysteresis Threshold | Gaussian Blur and Hysteresis Threshold|
|-------------------------|-------------------------|-----------------|
| <img src="https://user-images.githubusercontent.com/89137794/214329565-2d61ac82-3aab-42c8-98e5-25ee5dd98fa3.png" width="250"> | <img src="https://user-images.githubusercontent.com/89137794/214329571-3a19849a-1aea-4070-a795-0c15f5231909.png" width="250"> | <img src="https://user-images.githubusercontent.com/89137794/214329578-a54157ec-8e32-48e8-a61b-7a6589effd22.png" width="250">

### Hough Lines
The Hough Transform was effective at detecting straight edges in the image and would return a matrix representing all of the (𝜌,𝜃) values for each detected straight edge in the image which could represent a line. However it was not a very useful algorithm for use within this Graffiti image classifier because when applied to a test image, the detected straight edges did not have any length given to them so were represented with infinite length,. One of the reasons the straight edges within the images were being extracted was to compare the percentage of straight edges in the image to the number of edges in the image overall, therefore the straight-line edge detector needed to extract the straight edges along with their lengths

The Probabilistic Hough Transform was a better algorithm to use in this situation, it was a less computationally complex algorithm and it also returned straight edges along with their respective lengths.

|Original Image        |  Hough Transform | Probablistic Hough Transform|
|-------------------------|-------------------------|-----------------|
| <img src="https://user-images.githubusercontent.com/89137794/214317803-ded50f2f-3e66-4e20-acde-05e7c200e309.png" width="250"> | <img src="https://user-images.githubusercontent.com/89137794/214317767-92cc6700-0f30-4645-8eb0-fb7a16294da5.png" width="250"> | <img src="https://user-images.githubusercontent.com/89137794/214317836-87f1a253-d302-49f9-b762-bd321a517d4a.png" width="250">

## Neural Network

The different Computer Vision algorithms were used to extract numerical data from each of labelled images and the values were normalised between 0 and 1. Accuracy of the classifications were tested with different combinations of extracted features as neural networks as input. The neural Network was trained on 398 labelled images, 199 images of vandalism graffiti and 199 images of artwork graffiti. 15% of the images of the labelled data were not used to train the neural network, they were used to test the accuracy of the classifications so it would not cause overfitting. If a feature did not increase accuracy it was not included as an input to the final neural network graffiti image classifier.

## The Neural Network Architecture
A classification multilayer perceptron was created using the TensorFlow module in Python in order to use the extracted feature values from the labelled graffiti images to classify images into the two classes of vandalism graffiti and artwork graffiti. The multilayer perceptron architecture that was used had 5 layers, the first 4 hidden layers contained 10 neurons each and used a ReLU activation function, the output layer contained one neuron and used a sigmoid activation function. No momentum was used during training and the learning rate was 0.01. The numerical image feature values that were extracted from the labelled graffiti images that were used to train the multilayer perceptron were normalised between the values of 0 and 1 because a backpropagation gradient descent training algorithm was used.

<img src="https://user-images.githubusercontent.com/89137794/214282054-64f6d140-dd3e-4747-b44a-1042f0f87032.png">

## Final Neural Network Classification Accuracy

The Final Inputs for the Neural Network Graffiti Image Classifier were: Number of Regions, Region Colour Variance, Colourfullness (Hasler and Süsstrunk's Algorithm), Edge Orientation Variance, Number of Edges and Number of Straight Edges. The Accuracy of the trained neural network was 0.917.

Final Features Graphs        |  Final Features Confusion Matrix
:-------------------------:|:-------------------------:
<img src="https://user-images.githubusercontent.com/89137794/214330143-f0f3bdbd-da9b-4af0-808a-66e94b242a59.png" width="350"> |  <img src="https://user-images.githubusercontent.com/89137794/214331470-bbb88f41-66ca-4b01-91b6-08f68ca2fe44.png" width="350">

## Using the Classifier

Running the Application lets you select your own photos, the diffrent computer vision algorithms are then used on the selected photo to extract the numerical input values and these values are used as input into the trained neural network to classify your image into either Artwork Graffiti or Vandalism Graffiti. It allows you to view the different computer vision algorithms perfomed on your image and it gives you the numerical image data aquired from the computer vision algorithms which is used as input to the neural network classifier, both the actual values and the normalised values.

Artwork Examples | Vandalism Examples 
----------------|------------------
|<img src="https://user-images.githubusercontent.com/89137794/214331992-7e2f95c0-00c5-4d59-bbc4-8764f607e937.png" width= "350"> | <img src="https://user-images.githubusercontent.com/89137794/214331995-d292abf2-f5e6-412a-8461-49dd1d1defeb.png" width= "350"> |
 <img src="https://user-images.githubusercontent.com/89137794/214332000-9c8509e2-2e77-4c77-8e43-6c6f8e287a23.png" width= "350"> | <img src="https://user-images.githubusercontent.com/89137794/214331998-cba66ea3-b7fd-471a-9d11-14d6ad3a1555.png" width= "350"> |

