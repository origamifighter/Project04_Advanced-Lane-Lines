from moviepy.editor import VideoFileClip
from IPython.display import HTML

import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
from tracker import tracker

#Load previous camera calibration parameter
dist_pickle = pickle.load( open("./camera_cal/calibration_pickle.p","rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= sthresh[0])&(s_channel <= sthresh[1])] = 1
	
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hls[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= vthresh[0])&(v_channel <= vthresh[1])] = 1
	
	output = np.zeros_like(s_channel)
	output[(s_binary == 1) & (v_binary == 1)] = 1
	return output
	
	
images = glob.glob('./test_images/test*.jpg')

def process_image(img):
	#read and undistor images
	
	img = cv2.undistort(img,mtx,dist,None,mtx)
	
	
	# Process image and generate binary pixel of interests
	preprocessImage = np.zeros_like(img[:,:,0])
	gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
	grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
	c_binary = color_threshold(img, sthresh=(100,255),vthresh=(50,255))
	preprocessImage[((gradx == 1)&(grady == 1)|(c_binary == 1))] = 255
	
	
	
	# Defining perspective transformation area
	height = img.shape[0]
	width = img.shape[1]
	img_size = (width, height)

	
	top_left_src = (563, 470)
	bottom_left_src = (220, 700)
	top_left_dst = (300,300)
	bottom_left_dst = (300,720)
	
	src = np.float32([[top_left_src[0],top_left_src[1]], 
	[bottom_left_src[0],bottom_left_src[1]],
	[width - bottom_left_src[0],bottom_left_src[1]],
	[width - top_left_src[0],top_left_src[1]]]) 
	
	dst = np.float32([[top_left_dst[0],top_left_dst[1]], 
	[bottom_left_dst[0],bottom_left_dst[1]],
	[width - bottom_left_dst[0],bottom_left_dst[1]],
	[width - top_left_dst[0],top_left_dst[1]]])
	
	# Start applying perspective tranform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)
	
	warped_color = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
	
		
	
	#Define the box size for fitting the curvature line
	window_width = 25
	window_height = 80
	
	#set up the overeall class to do the tracking
	curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin =25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor=15)
	
	window_centroids = curve_centers.find_window_centroids(warped)
	
	# Points used to draw all the left and right window
	l_points = np.zeros_like(warped)
	r_points = np.zeros_like(warped)
	
	# Points used to find the left and right lanes
	leftx = []
	rightx = []
	
	# Go through each level and draw the windows 	
	for level in range(0,len(window_centroids)):
		# Window_mask is a function to draw window areas
		# Add center value found in frame to the list of lane points per left,right
		
		leftx.append(window_centroids[level][0])
		rightx.append(window_centroids[level][1])
		
		l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
		r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
		
		# Add graphic points from window mask here to total pixels found 
		l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
		r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
	
	
	# Draw the results
	#template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
	#zero_channel = np.zeros_like(template) # create a zero color channel
	#template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
	#warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
	#output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
	
	
	
	#Fit the lane boundaries to the left,right and center positions found
	yvals = range(0, warped.shape[0])
	
	res_yvals = np.arange(warped.shape[0]-(window_height/2),0,-window_height)
	
	left_fit = np.polyfit(res_yvals, leftx, 2)
	left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
	left_fitx = np.array(left_fitx,np.int32)
	
	right_fit = np.polyfit(res_yvals, rightx, 2)
	right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
	right_fitx = np.array(right_fitx,np.int32)
	
	left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2), axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
	
	road = np.zeros_like(img)
	road_bkg = np.zeros_like(img)
	cv2.fillPoly(road,[left_lane],color=[255,0,0])
	cv2.fillPoly(road,[right_lane],color=[0,0,255])
	cv2.fillPoly(road,[inner_lane],color=[0,100,0])
	#cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
	#cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])
	
	road_warped = cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
	road_warped_bkg = cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)
	
	base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
	final_line = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
	
	ym_per_pix = curve_centers.ym_per_pix
	xm_per_pix = curve_centers.xm_per_pix
	curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix,2)
	curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
	
	#Calculate the offset of the car on the road
	camera_center = (left_fitx[-1]+right_fitx[-1])/2
	center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
	side_pos = 'left'
	if center_diff <= 0:
		side_pos = 'right'
	
	cv2.putText(final_line, 'Radius of Curvature = '+str(round(curverad,3))+'(m)', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
	cv2.putText(final_line, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+side_pos+' of center', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
	
	return final_line
	
Output_video = 'output1_tracked.mp4'
Input_video = 'project_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_clip.write_videofile(Output_video, audio=False)	