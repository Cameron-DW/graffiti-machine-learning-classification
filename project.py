# Main Python Module for this Graffiti Image Classification Project

# Importing the necessary modules
import cv2
from to_database import *
from image_processing import Image
from NeuralNetwork import *
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
import PIL.Image
import PIL.ImageTk
from tkinter import font as tkfont
from DataSet import *
from sklearn.cluster import KMeans
import pandas as pd

data_file_load = 'data_extracted_image_feature_values.csv'
data_file_save = 'data_extracted_image_feature_values.csv'
min_val_load = 'min_val_image_feature_values.csv'
min_val_save = 'min_val_image_feature_values.csv'
neural_network_name = 'neural_network_classifier'
high_to_low_file = 'high_to_low_image_feature_values.csv'
icon_name = 'icon.ico'
user_file_path = ""


# Function to display an image
def show_image(image, name='image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)


# Function to normalise the extracted image features labelled graffiti images using the extracted feature values
# from the labelled training graffiti images
def normalise_input_data(input_data):
    input_data_transposed = np.transpose(input_data)

    features, instances = input_data_transposed.shape
    normalised_data = np.zeros((features, instances))

    list_max_feature_value = []
    list_min_feature_value = []
    for feature in input_data_transposed:
        max_feature_value = max(feature)
        list_max_feature_value.append(max_feature_value)
        min_feature_value = min(feature)
        list_min_feature_value.append(min_feature_value)

    high_to_low = [list_max_feature_value[i]-list_min_feature_value[i] for i in range(len(list_min_feature_value))]

    for feature in range(features):
        for instance in range(instances):
            normalised_data[feature, instance] = (input_data_transposed[feature, instance] - list_min_feature_value[
                feature]) / high_to_low[feature]

    print(high_to_low)
    print(list_min_feature_value)
    print(list_max_feature_value)

    save_max_feature_values(list_min_feature_value, high_to_low)

    normalised_data = np.transpose(normalised_data)

    return normalised_data


# Function to return the cluster centres from the SIFT descriptors
def k_means_sift(descriptors):
    #print(descriptors)
    descriptors = [item for sublist in descriptors for item in sublist]
    descriptors = np.array(descriptors)

    kmeans = KMeans(n_clusters=200).fit(descriptors)
    return kmeans.cluster_centers_


# Function to process the labelled graffiti images, takes all of the data_instances from the graffiti database and then
# turns these images into the Image Class, then from the values of the Image instance's attributes, a list is created,
# this list is appended to the array containing all of the feature values for the labelled training images. The same is
# done for the target values of these images and the normalised input values array is also created.
def process_images(data_instances):
    input_data = []
    target_data = []
    for data_instance in data_instances:
        print(f"Processing image: {data_instance.image_id}")
        print(data_instance.image_path)
        image = cv2.imread(data_instance.image_path)
        image_current = Image(image)
        sift_list = []

        classification = data_instance.get_classification()

        list_current_image_features = [image_current.num_regions,  # 1
                                       image_current.region_colour_variance,  # 2
                                       image_current.colourfulness_image,  # 3
                                       image_current.gradient_variance,  # 4
                                       image_current.lines_amount,  # 5
                                       image_current.edges_amount,  # 6
                                       image_current.region_size_variance,  # 7
                                       image_current.num_corners,  # 8
                                       image_current.straight_line_percent,  # 9
                                       image_current.average_red,  # 10
                                       image_current.average_green,  # 11
                                       image_current.average_blue,  # 12
                                       ]
        sift_list.append(image_current.sift_features)

        if classification == 'Vandalism':
            current_target = [1]
        else:
            current_target = [0]

        print(f"Values for extracted features: {list_current_image_features}\n")

        input_data.append(list_current_image_features)
        target_data.append(current_target)

    artwork_sift = k_means_sift(sift_list[:199])
    vandalism_sift = k_means_sift(sift_list[-199:])

    save_sift_centres(artwork_sift, vandalism_sift)

    input_data = np.array(input_data)
    input_data_normalised = normalise_input_data(input_data)
    target_data = np.array(target_data)

    return input_data, input_data_normalised, target_data


# the SIFT centres are saved for the training images so that they can be loaded at a later time
def save_sift_centres(artwork_sift_centres, vandalism_sift_centres):
    artwork_sift_filename = 'artwork_sift.csv'
    vandalism_sift_filename = 'vandalism_sift.csv'
    pd.DataFrame(artwork_sift_centres).to_csv(artwork_sift_filename, header=None, index=None)
    pd.DataFrame(vandalism_sift_centres).to_csv(vandalism_sift_filename, header=None, index=None)


def save_max_feature_values(min_feature_value_list, high_to_low):
    print(type(min_feature_value_list))
    print(type(high_to_low))
    pd.DataFrame(min_feature_value_list).to_csv(min_val_save, header=None, index=None)
    pd.DataFrame(high_to_low).to_csv(high_to_low_file, header=None, index=None)


def get_min_feature_values(file_path):
    file = open(file_path)
    list_min_feature_values= np.loadtxt(file, delimiter=",")
    return list_min_feature_values


def get_high_to_low_values(file_path):
    file = open(file_path)
    list_high_to_low_values= np.loadtxt(file, delimiter=",")
    return list_high_to_low_values


# input data and target data from the extracted
def save_data(input_data, target_data):
    input_data = np.transpose(input_data)
    target_data = np.transpose(target_data)

    total_data = np.concatenate((input_data, target_data))

    pd.DataFrame(total_data).to_csv(data_file_save, header=None, index=None)


# Main Function for creating the csv file used to train the NeuralNetwork object for graffiti image classification
def create_csv_training_data():

    # creating a random integer between value 1 and length of database
    db_length = length_of_database()
    data_instances = []

    # creating a list of all of the training data from the database as DataInstance objects
    for database_id in range(1, db_length+1):
        image_path = retrieve_image(database_id)
        classification = retrieve_classification(database_id)
        data_instances.append(DataInstance(database_id, image_path, classification))

    # processing all of the images from the data_instance list and
    # creating input data and target data to train the neural network
    input_data, input_data_normalised, target_data = process_images(data_instances)

    # saving the input and target data into a csv file
    save_data(input_data_normalised, target_data)


# Function to get the extracted image feature dataset from the labelled graffiti images
def get_training_data():
    data_original = pd.read_csv(data_file_load, header=None,)
    data = np.array(data_original)
    data = np.transpose(data)

    labelled_dataset = DataSet(data)
    print(labelled_dataset.array.shape)

    return labelled_dataset


# Function to create a NeuralNetwork object to classify the graffiti images
def create_neural_network(input_training_data, target_training_data):
    neural_net = NeuralNetwork(input_training_data, target_training_data)
    neural_net.setup()
    neural_net.train()

    # print the confusion matrix and the evaluation metrics (accuracy, precision, recall and F1 Score) of the neural
    # network classifier
    confusion_m = neural_net.get_confusion_matrix()
    print(confusion_m)
    neural_net.evaluation_metrics(confusion_m)

    return neural_net


# Function to save the neural network classifier so that it can be used with the GUI
def save_neural_network(neural_network, file_path):
    classifier = neural_network.classifier
    classifier.save(file_path)


# Function to load a neural network classifier so that it can be used with the GUI
def load_neural_network(file_path):
    classifier = keras.models.load_model(file_path)
    return classifier

# Function to normalise the user's inputted image for input into the classifer
def normalise_user_image(user_image):
    list_current_image_features =  [user_image.num_regions,
                                   user_image.region_colour_variance,
                                   user_image.colourfulness_image,
                                   user_image.gradient_variance,
                                   user_image.lines_amount,
                                   user_image.edges_amount,
                                   user_image.region_size_variance,
                                   user_image.num_corners,
                                   user_image.straight_line_percent,
                                   user_image.average_red,
                                   user_image.average_green,
                                   user_image.average_blue
                                   ]

    list_min_feature_values = get_min_feature_values(min_val_load)
    high_to_low = get_high_to_low_values(high_to_low_file)

    print(list_current_image_features)

    user_input = (list_current_image_features - list_min_feature_values) / high_to_low  # normalising it

    # removing the extracted image features not needed for input into classifer

    for i in range(len(user_input)):
        if user_input[i] > 1:
            user_input[i] = 1
        elif user_input[i] < 0:
            user_input[i] = 0
    print(user_input)
    user_input = user_input[0:6]
    number_of_inputs = user_input.shape[0]
    user_input = user_input.reshape(1, number_of_inputs)

    return user_input


# Function to create an Image object from the user's file path
def process_user_image(user_file_path):
    print("Processing Image")
    user_image = cv2.imread(user_file_path)
    user_image = Image(user_image)

    return user_image

# Function to determine the different image types from the user's image
def image_types(user_image):
    image_normal = user_image.image
    image_normal = cv2.cvtColor(image_normal, cv2.COLOR_BGR2RGB)
    image_segmented = user_image.image_segmented
    image_segmented = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB)
    image_edges = user_image.image_canny_edges
    image_edge_gradients = user_image.image_hog
    image_list = [image_normal, image_segmented, image_edges, image_edge_gradients]

    return image_list


# Function to create a list for different extracted image feature values for a user's image to be displayed on the GUI
def features_user_image(user_image):
    list_current_image_features = [user_image.num_regions,
                                   user_image.region_colour_variance,
                                   user_image.colourfulness_image,
                                   user_image.gradient_variance,
                                   user_image.lines_amount,
                                   user_image.edges_amount,
                                   user_image.region_size_variance,
                                   user_image.num_corners,
                                   user_image.straight_line_percent,
                                   user_image.average_red,
                                   user_image.average_green,
                                   user_image.average_blue
                                   ]

    feature_list = [round(i, 4) for i in list_current_image_features]

    label_image_features = f"Number of Corners: {feature_list[0]}\n" \
        f"Number of Segmented Regions: {feature_list[1]}\n" \
        f"Region Colour Variance: {feature_list[2]}\n" \
        f"Region Size Variance: {feature_list[3]}\n" \
        f"Image Colourfulness: {feature_list[4]}\n" \
        f"Edge Gradient Variance: {feature_list[5]}"

    return label_image_features

# Function to create a list for different NORMALISED extracted image feature values for a user's image to be displayed
# on the GUI
def features_user_image_normalised(user_image):
    list_current_image_features = [user_image.num_regions,
                                   user_image.region_colour_variance,
                                   user_image.colourfulness_image,
                                   user_image.gradient_variance,
                                   user_image.lines_amount,
                                   user_image.edges_amount,
                                   user_image.region_size_variance,
                                   user_image.num_corners,
                                   user_image.straight_line_percent,
                                   user_image.average_red,
                                   user_image.average_green,
                                   user_image.average_blue
                                   ]

    list_min_feature_values = get_min_feature_values(min_val_load)
    high_to_low = get_high_to_low_values(high_to_low_file)
    normalised_user_input = (list_current_image_features - list_min_feature_values)/high_to_low  # normalising it
    list_normalised_inputs = normalised_user_input.tolist()  # converting it to list

    feature_list_normalised = [round(i, 2) for i in list_normalised_inputs]

    for i in range(len(feature_list_normalised)):
        if feature_list_normalised[i] > 1:
            feature_list_normalised[i] = 1
        elif feature_list_normalised[i] < 0:
            feature_list_normalised[i] = 0

    label_image_features_normalised = f"Number of Regions: {feature_list_normalised[0]}\n" \
            f"Region Colour Variance: {feature_list_normalised[1]}\n" \
            f"Colourfulness: {feature_list_normalised[2]}\n" \
            f"Edge Orientation Variance: {feature_list_normalised[3]}\n" \
            f"Number of Straight Edges: {feature_list_normalised[4]}\n" \
            f"Number of Edges: {feature_list_normalised[5]}"

    return label_image_features_normalised


fln = ""
label_image_features_normalised = ""
label_image_features = ""
diff_images = [[], [], [], []]

# Function to create the GUI for the graffiti classifier using the module tkinter
def create_window():

    def classify():
        classifier = load_neural_network(neural_network_name)

        user_image = process_user_image(fln)
        global diff_images
        diff_images = image_types(user_image)

        global label_image_features
        label_image_features=features_user_image(user_image)

        normalised_user_input = normalise_user_image(user_image)
        print(normalised_user_input)

        global label_image_features_normalised
        label_image_features_normalised=features_user_image_normalised(user_image)

        class_estimate = classify_user_input(normalised_user_input, classifier)
        if class_estimate == 'Artwork Graffiti':
            text_colour = '#090'
        elif class_estimate == 'Vandalism Graffiti':
            text_colour = '#F00'

        lbl_classification.configure(text=f"Image classification: {class_estimate}", font=("Consolas", 18), fg=text_colour)
        print(class_estimate)

    def initialise_image():
        lbl_classification.configure(text="")
        global fln
        fln = filedialog.askopenfilename(title='Select Image File', filetypes=(('JPG File', '*.jpg'),))
        img = PIL.Image.open(fln)
        img = img.resize((512, 512))
        img = PIL.ImageTk.PhotoImage(img)
        lbl.configure(image=img)
        lbl.image = img

    def change_image_original():
        img = PIL.Image.fromarray(diff_images[0])
        img = img.resize((512, 512))
        img = PIL.ImageTk.PhotoImage(img)
        lbl.configure(image=img)
        lbl.image = img

    def change_image_segmented():
        img = PIL.Image.fromarray(diff_images[1])
        img = img.resize((512, 512))
        img = PIL.ImageTk.PhotoImage(img)
        lbl.configure(image=img)
        lbl.image = img

    def change_image_canny():
        img = PIL.Image.fromarray(diff_images[2])
        img = img.resize((512, 512))
        img = PIL.ImageTk.PhotoImage(img)
        lbl.configure(image=img)
        lbl.image = img

    def change_image_hog():
        img = PIL.Image.fromarray(np.uint8(diff_images[3]*255))  # converting it from 0-1 to 0-255
        img = img.resize((512, 512))
        img = PIL.ImageTk.PhotoImage(img)
        lbl.configure(image=img)
        lbl.image = img

    def view_information():
        info_window_root = tk.Tk()
        info_window_root.title("Image Feature Values")
        info_window_root.iconbitmap(icon_name)
        features_of_image_label = tk.Label(info_window_root)
        features_of_image_label.configure(text=label_image_features, font=default_font)
        features_of_image_label.pack()

        info_window_root.mainloop()

    def view_normalised_information():
        info_window_root_normalised = tk.Tk()
        info_window_root_normalised.title("Normalised Image Feature Values")
        info_window_root_normalised.iconbitmap(icon_name)
        features_of_image_label = tk.Label(info_window_root_normalised)
        features_of_image_label.configure(text=label_image_features_normalised, font=default_font)
        features_of_image_label.pack()

        info_window_root_normalised.mainloop()

    root = tk.Tk()
    frm = tk.Frame(root)
    root.iconbitmap(icon_name)

    frm.pack(side=tk.BOTTOM,padx=10, pady=10)
    lbl = tk.Label(root)
    lbl.pack()

    classification = tk.StringVar()
    lbl_classification = tk.Label(root)
    lbl_classification.pack(side=tk.BOTTOM, padx='30')

    default_font = tkfont.Font(family='consolas', size=10)

    btn = tk.Button(frm, text='Select Image', command=initialise_image, height=5, width=40, font=default_font)
    btn.pack(side=tk.LEFT)

    btn3 = tk.Button(frm, text='Classify Image', command=classify, height=5, width=40, font=default_font)
    btn3.pack(side=tk.LEFT, padx=10)

    root.title("Graffiti Classifier")
    root.geometry("580x690")

    my_menu = tk.Menu(root)
    root.config(menu=my_menu)

    view_menu = tk.Menu(my_menu)
    my_menu.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Original Image", command=change_image_original)
    view_menu.add_command(label="Segmented Image", command=change_image_segmented)
    view_menu.add_command(label="Edges of Image", command=change_image_canny)
    view_menu.add_command(label="Gradients of Image", command=change_image_hog)

    info_menu = tk.Menu(my_menu)
    my_menu.add_cascade(label="Info", menu=info_menu)
    info_menu.add_command(label="Features from Image", command = view_information)
    info_menu.add_command(label="Normalised Features from Image", command = view_normalised_information)

    root.mainloop()


# Function to classify the user input using the user's input and the trained neural network classifer
def classify_user_input(user_input, classifier):
    prediction = classifier.predict(user_input)
    if prediction >= 0.5:
        return "Vandalism Graffiti"
    else:
        return "Artwork Graffiti"


# Function to remove certain variables from the dataset so that the neural network is not trained on all extracted image
# feature values
def change_dataset(dataset):

    dataset.remove_variable(['region_size_var', 'num_corners',
                            'straight_line_percent', 'avg_green', 'avg_red', 'avg_blue',
                            'num_regions', 'region_colour_var', 'colourfulness', 'line_amount', 'edge_amount'])


def main():
    # Inserting the images into the database
    initial_insert()

    # Loading the images from the database, extracting features from them and saving into a csv file
    create_csv_training_data()

    # Loading the csv file into a DataSet object and then getting the input_data and the target_data
    dataset = get_training_data()
    change_dataset(dataset)
    print(dataset.variables)
    input_data = dataset.input()
    target_data = dataset.target()

    # Creating the neural network graffiti classifier and training it using the input and target data
    neural_network = create_neural_network(input_data, target_data)

    # Saving the neural network so it can be loaded and loaded on the GUI
    save_neural_network(neural_network, neural_network_name)

    # creating the GUI
    create_window()


main()





