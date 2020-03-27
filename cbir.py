from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import numpy as np
import cv2
import imutils
import csv


#creating the application main window.   
app = Tk()
app.title("Content based image retrival")
app.geometry("1300x600")
app.resizable(0, 0)



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

def open_image():
     global imagefile
     app.filename = filedialog.askopenfilename(initialdir="C:",title="Open image",filetypes=(("png files","*.png"),("all files","*.*")))
     img = ImageTk.PhotoImage(Image.open(app.filename))
     app.img = img
     photo.image = img
     photo.create_image(0, 0, anchor=NW, image=img)
     
def bengali():
    result = pytesseract.image_to_string(app.filename,lang ='ben')
    output.delete('1.0', END)
    output.insert(1.0,result)

def gujrati():
    result = pytesseract.image_to_string(app.filename,lang ='guj')
    output.delete('1.0', END)
    output.insert(1.0,result)

def hindi():
    result = pytesseract.image_to_string(app.filename,lang ='hin')
    output.delete('1.0', END)
    output.insert(1.0,result)

def kannada():
    result = pytesseract.image_to_string(app.filename,lang ='kan')
    output.delete('1.0', END)
    output.insert(1.0,result)

def malyalam():
    result = pytesseract.image_to_string(app.filename,lang ='hin')
    output.delete('1.0', END)
    output.insert(1.0,result)


def marathi():
    result = pytesseract.image_to_string(app.filename,lang ='mar')
    output.delete('1.0', END)
    output.insert(1.0,result)

def nepali():
    result = pytesseract.image_to_string(app.filename,lang ='nep')
    output.delete('1.0', END)
    output.insert(1.0,result)

def punjabi():
    result = pytesseract.image_to_string(app.filename,lang ='pun')
    output.delete('1.0', END)
    output.insert(1.0,result) 

def sanskrit():
    result = pytesseract.image_to_string(app.filename,lang ='san')
    output.delete('1.0', END)
    output.insert(1.0,result)

def sindhi():
    result = pytesseract.image_to_string(app.filename,lang ='snd')
    output.delete('1.0', END)
    output.insert(1.0,result)
    
def tamil():
    result = pytesseract.image_to_string(app.filename,lang ='tam')
    output.delete('1.0', END)
    output.insert(1.0,result)


#button defination 
button_0 = Button(app,text="Open Image",bg="slate gray",fg="black",command=open_image,font="Times",width=90)
button_1 = Button(app, text="Bengali",bg="ivory4",fg="white",width = "15",command=bengali,font="Times")
button_2 = Button(app, text="Gujrati",bg="ivory4",fg="white",width = "15",command=gujrati,font="Times")
button_3 = Button(app, text="Hindi",bg="ivory4",fg="white",width = "15",command=hindi,font="Times")
button_4 = Button(app, text="Kannada",bg="ivory4",fg="white",width = "15",command=kannada,font="Times")
button_5 = Button(app, text="Malyalam",bg="ivory4",fg="white",width = "15",command=malyalam,font="Times")
button_6 = Button(app, text="Marathi",bg="ivory4",fg="white",width = "15",command=marathi,font="Times")
button_7 = Button(app, text="Nepali",bg="ivory4",fg="white",width = "15",command=nepali,font="Times")
button_8 = Button(app, text="Punjabi",bg="ivory4",fg="white",width = "15",command=punjabi,font="Times")
button_9 = Button(app, text="Sanskrit",bg="ivory4",fg="white",width = "15",command=sanskrit,font="Times")
button_10 = Button(app, text="Sindhi",bg="ivory4",fg="white",width = "15",command=sindhi,font="Times")
button_11 = Button(app, text="Tamil",bg="ivory4",fg="white",width = "15",command=tamil,font="Times")


#canvas used to display image

photo = Canvas(app,bg="gray",width="400",height="200")

#label used to print output text;

output = Text(app,bg="LightYellow2",fg="black",)


#grid
button_0.grid(row=0,column=0,columnspan=3,padx=10,pady=5 )

photo.grid(row ="1",column="0",rowspan=11,padx="20")
output.grid(row=2,column = 2,rowspan=11,padx=10)


button_1.grid(row=1 ,column =1)
button_2.grid(row=2 ,column=1)
button_3.grid(row=3 ,column = 1)
button_4.grid(row=4 ,column = 1)
button_5.grid(row=5 ,column = 1)
button_6.grid(row=6 ,column = 1)
button_7.grid(row=7 ,column = 1)
button_8.grid(row=8 ,column = 1)
button_9.grid(row=9 ,column = 1)
button_10.grid(row=10 ,column = 1)
button_11.grid(row=11 ,column = 1)









#Entering the event main loop  
app.mainloop()  
