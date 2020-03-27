from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import numpy as np
import cv2
import imutils
import csv
import argparse
import glob



#creating the application main window.   
app = Tk()
app.title("Content based image retrival")
app.geometry("1300x600")
app.resizable(0, 0)



#================================================
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
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
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

#================================================
class Searcher:
	def __init__(self, indexPath):
		# store our index path
		self.indexPath = indexPath

	def search(self, queryFeatures, limit = 10):
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

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d

#================================================

# USAGE
# python index.py --dataset dataset --index index.csv

# import the necessary packages
#from pyimagesearch.colordescriptor import ColorDescriptor

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True,
#	help = "Path to the directory that contains the images to be indexed")
#ap.add_argument("-i", "--index", required = True,
#	help = "Path to where the computed index will be stored")
#args = vars(ap.parse_args())

# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# open the output index file for writing
output = open("C:\\Users\\Siddesh\\Desktop\\CBIR\\work\\index.csv","w")

# use glob to grab the image paths and loop over them
for imagePath in glob.glob("C:\\Users\\Siddesh\\Desktop\\CBIR\\work\\dataset" + "/*"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)

	# describe the image
	features = cd.describe(image)

	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file  ==================================================================It may can cause error
output.close()



def open_image():
     global imagefile
     app.filename = filedialog.askopenfilename(initialdir="C:",title="Open image",filetypes=(("png files","*.png"),("all files","*.*")))
     img = ImageTk.PhotoImage(Image.open(app.filename))
     app.img = img
     photo.image = img
     photo.create_image(0, 0, anchor=NW, image=img)
     


#button defination 
button_0 = Button(app,text="Open Image",bg="slate gray",fg="black",command=open_image,font="Times",width=90)


#canvas used to display image

photo = Canvas(app,bg="gray",width="400",height="200")

#label used to print output text;

output = Text(app,bg="LightYellow2",fg="black",)


#grid
button_0.grid(row=0,column=0,columnspan=3,padx=10,pady=5 )

photo.grid(row ="1",column="0",rowspan=11,padx="20")
output.grid(row=2,column = 2,rowspan=11,padx=10)


#Entering the event main loop  
app.mainloop()  
