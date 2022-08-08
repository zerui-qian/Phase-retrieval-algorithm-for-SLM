#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 22:54:31 2022

@author: zerui_qian
"""

# Need to download the Python module diffractio as in the link https://diffractio.readthedocs.io/en/latest/installation.html

from diffractio import mm, um, np, plt
from diffractio.scalar_masks_XY import Scalar_mask_XY
from PIL import Image
import os
os.chdir('/Users/zerui_qian/Desktop/UROP/1/Code') #Location of code

# Lens phase mask for SLM
M = 8*um            #pixel size of SLM in units of um
diameter=4000*M     #diameter of lens
focal=100*mm        #focal length

x0 = np.arange(0,1920)*M
y0 = np.arange(0,1200)*M
wl = 0.78 * um      #wavelength of source

t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wl)
t0.lens(r0=(960*M,600*M), radius=(diameter/2,diameter/2), focal=(focal,focal))

# Save lens phase mask
t0.save_mask('lens/lens100.bmp', kind='phase')

lens = Image.open('lens/lens100.bmp').convert('L')
lens = np.asarray(lens,float)
lens = 256*np.ones((1200,1920)) - np.asarray(lens, float)
plt.imshow(lens, cmap='gray')
plt.show()

lens = Image.fromarray(lens).convert('L')
lens.save('lens/lens100.bmp')    #File name: lens(focal length)