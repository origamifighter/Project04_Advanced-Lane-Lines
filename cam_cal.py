import numpy as np
import cv2
import glob
import pickle

#Prepare object points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

#Arrays to store object points and image points from all the images

objpoints = []
imgpoints = []

images = glob.glob('./camera_cal/calibration*.jpg')


for idx, fname in enumerate(images):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#Find the checkboard corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	
	
	if ret == True:
		print ('working on ', fname)
		objpoints.append(objp)
		imgpoints.append(corners)
		
		#Draw and Display the corners
		cv2.drawChessboardCorners(img,(9,6), corners, ret)
		write_name = './camera_cal/corners_found'+str(idx)+'.jpg'
		cv2.imwrite(write_name, img)

#Load image for reference
img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1],img.shape[0])
print(img_size)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

#Save camera calibration for later use
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open("./camera_cal/calibration_pickle.p", "wb"))

images_undistorted = glob.glob('./camera_cal/calibration*.jpg')

for idx, fname in enumerate(images_undistorted):
	#read and undistort images
	img = cv2.imread(fname)
	img = cv2.undistort(img,mtx,dist,None,mtx)
	
	write_name = './camera_cal/undistort'+str(idx+1)+'.jpg'
	cv2.imwrite(write_name, img)
