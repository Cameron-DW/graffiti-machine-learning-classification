# Python Module to create the database for the labelled graffiti images and to allow for these images to be inserted
# into the database, retrieved from the database, etc.

import pymysql.connections
import os
import math

folder_name = "labelled_graffiti_images"

myDB = pymysql.connect(
    host="localhost",
    user="root",
    password="admin",
    database="graffiti_db"
)

MyCursor = myDB.cursor()

MyCursor.execute("CREATE TABLE IF NOT EXISTS Images( "
                 "id INTEGER(5) PRIMARY KEY,"
                 " Photo LONGBLOB NOT NULL, Classification CHAR(1)"
                 ")"
                 )


# Function to get the length of the database
def length_of_database():
    sql_statement = "SELECT COUNT(id) FROM Images"
    MyCursor.execute(sql_statement)
    length = MyCursor.fetchone()[0]
    return length


# Function to get an image
def retrieve_image(database_id):
    sql_statement = f"SELECT * FROM Images WHERE id = '{database_id}'"
    MyCursor.execute(sql_statement)
    my_result = MyCursor.fetchone()[1]
    store_file_path = f"DatabaseImages/img{str(database_id)}.jpg"
    with open(store_file_path, "wb") as File:
        File.write(my_result)
        File.close()

    return store_file_path


# Function to get the classification of an image
def retrieve_classification(database_id):
    sql_statement = f"SELECT Classification FROM Images WHERE id = '{database_id}'"
    MyCursor.execute(sql_statement)
    classification = MyCursor.fetchone()[0]
    return classification


# Function to insert an item into the database
def insert_item(initial_id, file_path, classification):
    with open(file_path, "rb") as File:
        binary_data_of_image = File.read()
    sql_statement = f"INSERT INTO Images (id, Photo, Classification) VALUES (%s,%s,%s)"
    MyCursor.execute(sql_statement, (initial_id, binary_data_of_image, classification))
    myDB.commit()


# Function to show the percentage of the upload of the graffiti images into the database
def percentage(records_inserted, total_records):
    db_percentage_complete = math.floor(records_inserted / total_records * 100)
    percent_string = f"upload is {db_percentage_complete}% complete"
    print(percent_string)


# Function to insert the initial labelled graffiti images from the directory folder into the database
def initial_insert():
    MyCursor.execute("DELETE FROM Images")
    directory = fr'C:\Users\devan\Documents\University\Year 3\Project\graffiti\{folder_name}'
    records_inserted = 0
    total_records = len(os.listdir(directory))

    for file_name in os.listdir(directory):
        initial_id = records_inserted + 1
        file_path = f"{folder_name}/{file_name}"
        classification = file_name[0]
        insert_item(initial_id, file_path, classification)
        records_inserted += 1
        percentage(records_inserted, total_records)


# A DataInstance class that the database will contain an ID, image path and classification for a graffiti image
class DataInstance:
    def __init__(self, image_id, image_path, classification):
        self.image_id = image_id
        self.image_path = image_path
        self.classification = classification

    def __str__(self):
        return f"({self.image_id}, {self.image_path}, {self.classification})"

    def get_classification(self):
        if self.classification == "g":
            return "Graffiti"
        elif self.classification == "v":
            return "Vandalism"
