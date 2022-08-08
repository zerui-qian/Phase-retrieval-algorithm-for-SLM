#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 22:03:27 2022

@author: tonyqian
"""

import numpy as np

def Ger_Sax_algo(beam, img, pm_s, h, w, max_iter=100):
    '''
    Gerchberg-Saxton algorithm

    Parameters
    ----------
    beam : 2darray
        Image of the input beam captured by a detector.
    img : 2darray
        Desired image.
    pm_s : 2darray
        Initial phase mask at the SLM.
    h : integer
        Height of SLM in units of pixels.
    w : integer
        Width of SLM in units of pixels.
    max_iter : integer, optional
        Number of iterations. The default is 100.

    Returns
    -------
    Phase mask for the desired image

    '''
    pm_f = np.random.rand(h, w)     #phase mask at the focal plane
    am_s = beam                     #amplitude of input beam before being modified by SLM
    am_f = np.sqrt(img)             #amplitude of desired image

    signal_s = am_s * np.exp(pm_s * 1j) #Beam after being modified by pm_s

    for iter in range(max_iter):
        signal_f = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(signal_s)))  #Propagate modified beam to focal plane
        pm_f = np.angle(signal_f)                                           #New phase mask at focal plane
        signal_f = am_f * np.exp(pm_f * 1j)                                 #Update amplitude at focal plane with the desired image
        signal_s = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(signal_f))) #Propagate back to the SLM plane
        pm_s = np.angle(signal_s)                                           #New phase mask at SLM
        signal_s = am_s * np.exp(pm_s * 1j)                                 #Update phase mask at SLM plane
    
    return pm_s

def twoD_Gaussian(data, A, xo, yo, sigma_x, sigma_y, theta=0, offset=0):
    '''
    Two dimensional Gaussian distribution

    Parameters
    ----------
    data : tuple
        meshgrid of xy coordinate
    A : float
        Ampltiude of Gaussian distribution
    xo : float
        x coordinate of centre.
    yo : float
        y coordinate of centre.
    sigma_x : float
        Width in x direction.
    sigma_y : float
        Width in y direction.
    theta : float, optional
        Angle of inclination to the xy plane. The default is 0.
    offset : float, optional
        A constant added to the 2d Gaussian function. The default is 0.

    Returns
    -------
    TYPE
        2darray 

    '''
    (x, y) = data                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + A*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()