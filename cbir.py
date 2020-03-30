# for csv operation
import csv
# to access images in folder
import glob
import tkinter as tk
# for GUI
from tkinter import *
# for file dialog
from tkinter import filedialog

# for image operations
import cv2
# for basic image processing
import imutils
# for image operations to arrays
import numpy as np
# image operations
from PIL import ImageTk, Image



class ColorDescriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),(0, cX, cY, h)]

        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        # return the feature vector
        return features

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])

        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()

        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()

        # return the histogram
        return hist


class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=10):
        # initialize our dictionary of results
        results = {}

        # open the index file for reading
        with open(self.indexPath) as f:
            # initialize the CSV reader
            reader = csv.reader(f)

            # loop over the rows in the index
            for row in reader:
                # parse out the image ID and features, then compute the
                # chi-squared distance between the features in our index
                # and our query features
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)

                # now that we have the distance between the two feature
                # vectors, we can udpate the results dictionary -- the
                # key is the current image ID in the index and the
                # value is the distance we just computed, representing
                # how 'similar' the image in the index is to our query
                results[row[0]] = d

            # close the reader
            f.close()

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])

        # return the chi-squared distance
        return d


def open_folder():
    global folder_name
    global result_data
    global current_index
    result_data = []
    current_index = 0
    # To select dataset folder
    folder_name = filedialog.askdirectory()
    cd = ColorDescriptor((8, 12, 3))
    output = open(folder_name + "/index.csv", "w")  # creating csv file
    filecount = 1
    for imagePath in glob.glob(folder_name + "//" "*.*"):
        if imagePath.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # printing indexed images
            text_output.insert(1.0, str(filecount) + " : " + imagePath + "\n")
            filecount += 1
            imageID = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            features = cd.describe(image)
            # write the features to file
            features = [str(f) for f in features]
            output.write(imagePath + "," + ",".join(features) + "\n")
    text_output.insert(1.0, "Indexed images : \n")
    output.close()


def select_image():
    global image_file
    global result_data
    result_data = []
    # selecting image to search
    image_file = filedialog.askopenfilename(initialdir="C:", title="Open image",
                                            filetypes=(("jpeg files", "*.jpg"),("png files", "*.png"), ("all files", "*.*")))
    # adding image to result file
    result_data.append(image_file)
    text_output.delete(1.0, END)
    # printing similar images on console
    text_output.insert(1.0, image_file)
    text_output.insert(1.0, "Selected image : ")

    width = 400
    height = 400
    img = Image.open(image_file)
    img = img.resize((width, height), Image.ANTIALIAS)
    img1 = ImageTk.PhotoImage(img)
    image_panel.configure(image=img1)
    image_panel.image = img1


def search_similar():
    global image_file
    global result_data
    global folder_name
    global current_index

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))

    current_index = 0
    text_output.delete(1.0, END)

    # load the query image and describe it
    query = cv2.imread(image_file)
    features = cd.describe(query)

    # perform the search
    searcher = Searcher(folder_name + "/index.csv")
    results = searcher.search(features)

    if (len(results) == 0):
        text_output.delete(1.0, END)
        text_output.insert(1.0, "No similar images found...")
    else:
        count = 1
        for (score, resultID) in results:
            # adding reults to the album
            result_data.append(resultID)
            text_output.insert(1.0, str(count) + " " + resultID + "\n")
            count += 1

        width = 400
        height = 400
        img = Image.open(result_data[0])
        img = img.resize((width, height), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img)
        image_panel.configure(image=img1)
        image_panel.image = img1
    current_index = 1
    text_output.insert(1.0, "Total " + str(len(result_data) - 1) + " similar images found \n ")
    display()


def display():
    # function to display similar images
    width = 400
    height = 400
    if (len(result_data) > 1):
        # text_output.delete(1.0, END)
        # text_output.insert(1.0, str(len(result_data)))

        img = Image.open(result_data[current_index])
        img = img.resize((width, height), Image.ANTIALIAS)
        img1 = ImageTk.PhotoImage(img)
        image_panel.configure(image=img1)
        image_panel.image = img1


def position():
    # for result check
    if len(result_data) != 0:
        return TRUE
    else:
        return FALSE


def next():
    # function to display next image
    global current_index
    if position() and (current_index + 1 != len(result_data)):
        current_index += 1
        display()
    else:
        pass


def back():
    # function to display previous image
    global current_index
    if position() and current_index != 0:
        current_index -= 1
        display()
    else:
        pass



app = Tk()
app.geometry("1200x600")

result_data = []
current_index = 0
app.title("content based image retrieval")
app.resizable(0, 0)

# frame to store button
button_frame = tk.Frame(app, relief=RIDGE, width=1100, height=50, borderwidth=3)
button_frame.pack(side=TOP)
button_frame.pack_propagate(0)

folder_button = Button(button_frame, text=" select dataset [Image Folder] ", command=open_folder, relief=RIDGE,
                       bg="white")
folder_button.pack(side=LEFT)

image_button = Button(button_frame, text=" select image ", command=select_image, relief=RIDGE, bg="white")
image_button.pack(side=LEFT)

search_button = Button(button_frame, text=" search ", command=search_similar, relief=RIDGE, bg="white")
search_button.pack(side=LEFT)

# frame to display image meta data
result_frame = tk.Frame(app, relief=RIDGE, width=400, height=500, borderwidth=3, )
result_frame.pack(side=LEFT)
result_frame.pack_propagate(0)

text_output = Text(result_frame)
text_output.pack(expand=True, fill='both')

image_output = tk.Frame(app, relief=RIDGE, width=800, height=500, borderwidth=3)
image_output.pack(side=LEFT)
image_output.pack_propagate(0)

image_panel = Label(image_output)
image_panel.pack(side=TOP)

# frame to display similar images
album_frame = tk.Frame(image_output, relief=RIDGE, width=800, height=50, borderwidth=3)
album_frame.pack(side=BOTTOM)

back_button = Button(album_frame, text="<<", command=back)
back_button.pack(side=LEFT)

next_button = Button(album_frame, text=">>", command=next)
next_button.pack(side=RIGHT)

app.mainloop()
