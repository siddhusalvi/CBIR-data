import csv
import glob
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import  time
import cv2
import imutils
import numpy as np


# ==========================================================================================================================================================
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
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                    (0, cX, cY, h)]

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
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])

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


# ==========================================================================================================================================================

def open_image():
    global resultdata
    global filename
    filename = filedialog.askdirectory()

    cd = ColorDescriptor((8, 12, 3))
    output = open(filename + "/index.csv", "w")
    # use glob to grab the image paths and loop over them
    filecount = 1
    for imagePath in glob.glob(filename +"//" "*.*"):
        if imagePath.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            text.insert(1.0, str(filecount) + " : " + imagePath + "\n")
            filecount += 1
            imageID = imagePath[imagePath.rfind("/") + 1:]
            image = cv2.imread(imagePath)
            window = "data"
            image = cv2.resize(image, (500, 300))
            # cv2.imshow(window,image)
            # cv2.waitKey(0)
            # describe the image
            features = cd.describe(image)

            # write the features to file
            features = [str(f) for f in features]
            output.write(imagePath + "," + ",".join(features) + "\n")
    text.insert(1.0, "Indexed images : \n")

    output.close()


def select_image():
    global fileimage
    fileimage = filedialog.askopenfilename(initialdir="C:", title="Open image",filetypes=(("jpeg files","*.jpg"),("all files","*.*")))
    resultdata.append(fileimage)
    text.delete(1.0, END)
    text.insert(1.0, fileimage)
    text.insert(1.0, "Selected image : ")

    width = 400
    height = 400
    img = Image.open(fileimage)
    img = img.resize((width, height), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img)
    panel.configure(image=img2)
    panel.image = img2


    # load the query image and describe it
def display():
    width = 400
    height = 400
    img = Image.open(resultdata[current])
    img = img.resize((width, height), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img)
    panel.configure(image=img2)
    panel.image = img2
def position():
    if len(resultdata) != 0:
        return TRUE
    else:
        return FALSE
def next():
    global current
    if(current+1 == len(resultdata)) and position():
        pass
    else:
        current+=1
        display()
def back():
    global current
    if current == 0 and position():
        pass
    else:
        current -=1
        display()

def search_similar():
    # perform the search
    # print(filename)

    global fileimage
    global resultdata
    global filename

    # initialize the image descriptor
    cd = ColorDescriptor((8, 12, 3))

    # load the query image and describe it
    query = cv2.imread(fileimage)
    features = cd.describe(query)

    # perform the search
    searcher = Searcher(filename+"/index.csv")
    results = searcher.search(features)

    if(len(results)==0):
        text.delete(1.0, END)
        text.insert(1.0, "No similar images found...")
    else:
        count = 1
        for (score, resultID) in results:
            # load the result image and display it
            resultdata.append(resultID)
            text.insert(1.0, str(count) + " " + resultID + "\n")
            count += 1
        text.insert(1.0, "Similar images : " + "\n")

        width = 400
        height = 400
        print(resultdata[0])
        img = Image.open(resultdata[0])
        img = img.resize((width, height), Image.ANTIALIAS)
        img2 = ImageTk.PhotoImage(img)
        panel.configure(image=img2)
        panel.image = img2
    for i in resultdata:
        print(i)
    print(len(resultdata))
# ==========================================================================================================================================================


app = Tk()
app.geometry("1200x600")

resultdata = []
current = 0
app.title("Content based Image Retrieval")
app.resizable(0, 0)

my_frame3 = tk.Frame(app, relief=RIDGE, width=1100, height=50, borderwidth=3)
my_frame3.pack(side=TOP)
my_frame3.pack_propagate(0)

index = Button(my_frame3, text=" index data ", command=open_image, relief=RIDGE, bg="white")
index.pack(side=LEFT)

image = Button(my_frame3, text=" select image ", command=select_image, relief=RIDGE, bg="white")
image.pack(side=LEFT)

search = Button(my_frame3, text=" search ", command=search_similar, relief=RIDGE, bg="white")
search.pack(side=LEFT)

my_frame = tk.Frame(app, relief=RIDGE, width=400, height=500, borderwidth=3, )
my_frame.pack(side=LEFT)
my_frame.pack_propagate(0)

text = Text(my_frame)
text.pack(expand=True, fill='both')


my_frame1 = tk.Frame(app, relief=RIDGE, width=800, height=500, borderwidth=3)
my_frame1.pack(side=LEFT)
my_frame1.pack_propagate(0)

panel = Label(my_frame1,bg="green")
panel.pack(side = TOP)


my_frame4 = tk.Frame(my_frame1, relief=RIDGE, width=800, height=50, borderwidth=3)
my_frame4.pack(side=BOTTOM)

back = Button(my_frame4,text="<<",command=back)

back.pack(side = LEFT)

next =Button(my_frame4,text=">>",command=next)
next.pack(side = RIGHT)






app.mainloop()
