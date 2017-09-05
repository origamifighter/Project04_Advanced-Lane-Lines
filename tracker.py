import numpy as np
import cv2

class tracker():

	def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
	
		self.recent_centers = []
		
		self.window_width = Mywindow_width
		self.window_height = Mywindow_height
		self.margin = Mymargin
		self.ym_per_pix = My_ym
		self.xm_per_pix = My_xm
		self.smooth_factor = Mysmooth_factor
		
	def find_window_centroids(self, warped):
		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin
		
		window_centroids = []
		window = np.ones(window_width)
		
		# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
		# and then np.convolve the vertical image slice with the window template 
		
		# Sum quarter bottom of image to get slice, could use a different ratio
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
		
		# Add what we found for the first layer
		window_centroids.append((l_center,r_center))
		
		# Go through each layer looking for max pixel locations
		for level in range(1,(int)(warped.shape[0]/window_height)):
			# convolve the window into the vertical slice of the image
			image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
			conv_signal = np.convolve(window, image_layer)
			# Find the best left centroid by using past left center as a reference
			# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
			offset = window_width/2
			l_min_index = int(max(l_center+offset-margin,0))
			l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
			# Find the best right centroid by using past right center as a reference
			r_min_index = int(max(r_center+offset-margin,0))
			r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
			# Add what we found for that layer
			window_centroids.append((l_center,r_center))

		self.recent_centers.append(window_centroids)
		
		return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)