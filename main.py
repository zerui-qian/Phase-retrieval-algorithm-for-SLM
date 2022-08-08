#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 22:05:07 2022

@author: tonyqian
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from PIL import Image
import os
os.chdir('/Users/zerui_qian/Desktop/UROP/1/Code') #Location of code
from phase_retrieval import *

#%%
'''
Initialisation
'''
#%% Reads the input beam image captured by beam profiler
beam = Image.open('Beam_profiler.bmp').convert('L')
beam = np.asarray(beam,float)

slm_size_x, slm_size_y = 15.36, 9.6             #slm dimension in mm
det_size_x, det_size_y = 11.264, 11.264         #detector dimension in mm
det_res_x, det_res_y = 2048, 2048               #detector resolution in pixels

#Dimension of SLM in terms of detector's resolution
y = int(slm_size_y / det_size_y * det_res_y)
x = int(slm_size_x / det_size_x * det_res_x)

#Crop the image so that it fits in SLM
mx, my = 1024, 1024
beam = beam[int(my - y/2):int(my + y/2),:]

#Extend the image with ones so that it fits in SLM
ext = int((x - 2048)/2)
beam = np.c_[np.ones((y,ext)), beam, np.ones((y,ext))]

#Change resolution of image to that of the SLM
beam = Image.fromarray(beam)
beam = beam.resize((1920,1200))
beam = np.asarray(beam,float)

plt.imshow(beam, cmap='gray')
plt.title('Input beam')
plt.show()

#%% Reads the input beam image captured by the Li camera
beam = Image.open('Beam_li_cam.bmp').convert('L')
beam = np.asarray(beam,float)

slm_size_x, slm_size_y = 15.36, 9.6                 #slm dimension in mm
det_size_x, det_size_y = 0.64*9.9, 0.48*9.9       #detector dimension in mm
det_res_x, det_res_y = 640, 480                     #detector resolution in pixels

#Dimension of SLM in terms of detector's resolution
y = int(slm_size_y / det_size_y * det_res_y)
x = int(slm_size_x / det_size_x * det_res_x)

#Extend the image with ones so that it fits in SLM
ext_x = int((x - 640)/2)
ext_y = int((y - 480)/2)
beam = np.c_[np.ones((480,ext_x)), beam, np.ones((480,ext_x))]
beam = np.r_['0,2', np.ones((ext_y,640+2*ext_x)), beam, np.ones((ext_y,640+2*ext_x))]

#Change resolution of image to that of the SLM
beam = Image.fromarray(beam)
beam = beam.resize((1920,1200))
beam = np.asarray(beam,float)

plt.imshow(beam, cmap='gray')
plt.title('Input beam')
plt.show()

#%% Remove elongation in actual image
x = np.arange(0,640)
y = np.arange(0,480)
x, y = np.meshgrid(x, y)
z = Image.open('Examples/1.bmp').convert('L')
z = np.asarray(z,float)

#Initial guesses of parameters of the four spots
#(amplitude, mu_x, mu_y, sigma_x, sigma_y)
g1 = (256,340,50,10,10,0,0)
g2 = (256,340,420,10,10,0,0)
g3 = (256,230,240,10,10,0,0)
g4 = (256,460,240,10,10,0,0)
xdata = np.vstack((x.ravel(),y.ravel()))
ydata = z.ravel()

#Fits each of the four spots to a 2D Gaussian distribution
popt1, pcov1 = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=g1)
data_fitted_1 = twoD_Gaussian((x, y), *popt1).reshape(480,640)

popt2, pcov2 = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=g2)
data_fitted_2 = twoD_Gaussian((x, y), *popt2).reshape(480,640)

popt3, pcov3 = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=g3)
data_fitted_3 = twoD_Gaussian((x, y), *popt3).reshape(480,640)

popt4, pcov4 = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=g4)
data_fitted_4 = twoD_Gaussian((x, y), *popt4).reshape(480,640)

#Plots the four 2D Gaussian distributions on top of the image
fig, ax = plt.subplots(1, 1)

ax.imshow(z, cmap='gray')
ax.contour(x, y, data_fitted_1, 5, cmap='RdGy')
ax.contour(x, y, data_fitted_2, 5, cmap='RdGy')
ax.contour(x, y, data_fitted_3, 5, cmap='RdGy')
ax.contour(x, y, data_fitted_4, 5, cmap='RdGy')
plt.show()

#Calculates the height and width of the rhombus and finds the aspect ratio
h = np.sqrt((popt1[1]-popt2[1])**2 + (popt1[2]-popt2[2])**2)
w = np.sqrt((popt3[1]-popt4[1])**2 + (popt3[2]-popt4[2])**2)

c = h/w     #aspect ratio

#%%
"""
Shape of desired image
"""
#%% Rhombus
img = np.zeros((1200,1920))
R = 100     #Half of the diagonal length

img[600 - int(R/c), 960] = 256
img[600 + int(R/c), 960] = 256
img[600, 960 - R] = 256
img[600, 960 + R] = 256

plt.imshow(img, cmap='gray')
plt.title('Desired image')
plt.ylim(550,650)
plt.xlim(900,1020)
plt.show()

#%% Square image
img = np.zeros((1200,1920))

W, H = 20,20               #Number of rows (H) and columns (W)
w, h = 20,int(20/c)        #Distance between rows (h) and columns (w)
for i in np.arange(0, W):
    for j in np.arange(0, H):
        img[600 - int((H-1)/2*h) + j*h, 960 + int((W-1)/2*w) - i*w] = 256

plt.imshow(img, cmap='gray')
plt.title('Desired image')
plt.ylim(450,750)
plt.xlim(800,1120)
plt.show()

#%% Circle image
img = np.zeros((1200,1920))
h, w = img.shape[:2]        #height and width of image
r = 100                     #radius of circle
for i in range(h):
    for j in range(w):
        if np.sqrt((i-600)**2+(j-960)**2) < r:
            img[i,j] = 256
plt.imshow(img, cmap='gray')
plt.show()

#%% Gaussian image
x = np.arange(0,1920)
y = np.arange(0,1200)
sigma_x, sigma_y = 10,10    #Gaussian widths
mu_x, mu_y = 960, 600       #Centre

x, y = np.meshgrid(x, y)
img = 256 * np.exp(-((x-mu_x)**2/(2*sigma_x**2)
     + (y-mu_y)**2/(2*sigma_y**2)))

plt.imshow(img, cmap='gray')
plt.ylim(550,650)
plt.xlim(900,1020)
plt.title('Desired image')
plt.show()

#%%
"""
Calculate Phase mask
"""
#%% Calculates the phase mask of the desired image given the input beam and simulates the image at the focal plane
phase_mask_0 = Ger_Sax_algo(beam, img, np.zeros((1200, 1920)), 1200, 1920, 100)     #Starts with a pahse mask of 0s (can be changed)
modified_beam = beam * np.exp(phase_mask_0 * 1j)    #beam after being modified by the slm
recovery = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(modified_beam)))

pm = np.uint8(((phase_mask_0/(2*np.pi))*256)+128)   #converts range of phase mask to 256
plt.imshow(pm, cmap='gray')
plt.title('Phase mask')
plt.show()

plt.imshow(np.absolute(recovery)**2, cmap='gray')
#plt.ylim(550,650)
#plt.xlim(900,1020)
plt.title('Recovered image')
plt.show()

#%% add a virtual lens to the phase mask
lens = Image.open('lens/lens200.bmp').convert('L')
lens = np.asarray(lens,float)
lens = lens/256 * 2 * np.pi     #convert range of phase mask to 2pi

#Superimposes phase mask of the lens onto that of the desired image
modified_beam = beam * np.exp(phase_mask_0 * 1j) * np.exp(lens * 1j)
pm = np.angle(modified_beam)
pm = np.uint8(pm/(2*np.pi)*256 + 128)   #converts range of phase mask to 256

plt.imshow(pm, cmap='gray')
plt.title('Phase mask')
plt.show()

pm = Image.fromarray(pm)
pm.save("Grid/Phi/Phi001.bmp")

#%% Behaviour of GS algorithm for different number of iterations
# r = []
# n = np.arange(0,201,5)
# for i in n:
#     phase_mask = Ger_Sax_algo(beam, img, i)
#     recovery = np.fft.fft2(np.exp(phase_mask * 1j))
    
#     a1 = img
#     a2 = np.absolute(recovery)**2
#     r.append(np.corrcoef(a1.flat, a2.flat)[0, 1])

# plt.title('Convergence')
# plt.xlabel('iterations')
# plt.ylabel('correlation coefficient')
# plt.plot(n,r,'k.')
# plt.show()

#%% 2d Gaussian fit to a spot
x = np.arange(0,140)
y = np.arange(0,140)
x, y = np.meshgrid(x, y)
z = Image.open('Examples/4.bmp').convert('L')[180:320,260:400]
z = np.asarray(z,float)

initial_guess = (256,70,70,10,10,0,0)   #(amplitude, mu_x, mu_y, sigma_x, sigma_y)

#Fitting the function to the image
xdata = np.vstack((x.ravel(),y.ravel()))
ydata = z.ravel()
popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, ydata, p0=initial_guess)
data_fitted = twoD_Gaussian((x, y), *popt).reshape(140,140)

fig, ax = plt.subplots(1, 1)
ax.imshow(z, cmap='gray')
ax.contour(x, y, data_fitted, 10, cmap='RdGy')
plt.show()

#Correlation coefficient between image and fitted function
r = np.corrcoef(data_fitted.flat, z.flat)[0, 1]
print(r)

#%%
'''
Equalise trap intensities with feedback loop
'''
#%% Initialize phase mask to the previous one
phase_mask = phase_mask_0
#%%
from skimage.feature import peak_local_max
img_0 = img                                             #Desired image
img_1 = Image.open('Grid/Img001.bmp').convert('L')      #Previous image captured by camera in the nth iteration
img_1 = np.asarray(img_1,float)
plt.imshow(img_1, cmap='hot')
plt.show()

#Finds local maxima in the desired and captured image given the number of spots (num_peaks)
xy_0 = peak_local_max(img_0, min_distance=2, num_peaks=400)
xy_1 = peak_local_max(img_1, min_distance=2, num_peaks=400)
In_0 = img_0[xy_0[:,0],xy_0[:,1]]
In_1 = img_1[xy_1[:,0],xy_1[:,1]]
In_1_mean = np.mean(In_1)

#Plots histogram of captured image
plt.hist(In_1, np.arange(0,255,5))
plt.xlim(0,255)
plt.ylim(0,400)
plt.show()

#Updates the desired image 
G = 0.7     #G-factor
In_2 = In_1_mean/(1 - G*(1-In_1/In_1_mean))

img_2 = img_0
img_2[xy_0[:,0],xy_0[:,1]] = In_2

#%%
# x= np.arange(0,400)
# plt.plot(x, In_1)
# plt.plot(x, In_2)
# plt.ylim(0,100)
#%% Finds the new phase mask given the updated desired image
phase_mask = Ger_Sax_algo(beam, img_2, phase_mask, 1200, 1920, 20)

modified_beam = beam * np.exp(phase_mask * 1j) * np.exp(lens * 1j)
pm = np.angle(modified_beam)
pm = np.uint8(pm/(2*np.pi)*256 + 128)

plt.imshow(pm, cmap='gray')
plt.title('Phase mask')
plt.show()

pm = Image.fromarray(pm)
pm.save("Grid/Phi/Phi002.bmp")  #Change the file name to Phi00(n+1) for the nth iteration
#Then use the camera to take the image, save it as Img00(n+1) and restart the feedback loop without setting phase_mask = phase_mask_0