import sys
sys.path.append("/home/chemisteyeirtwo/chemist_eye/lib/python3.11/site-packages")
sys.path.append("/home/chemisteyeirtwo/pysenxor-master")
import os
import signal
import time
import serial
import numpy as np
import cv2 as cv
import requests
import base64

from senxor.mi48 import MI48
from senxor.utils import data_to_frame, remap, cv_filter,\
                         cv_render, RollingAverageFilter,\
                         connect_senxor

# Global MI48 instance
global mi48
list_ironbow_b = [0,6,12,18,27,38,49,59,64,68,73,78,82,86,90,94,98,102,105,109,112,115,119,122,124,127,129,132,134,136,138,140,142,145,147,148,150,151,152,153,154,155,157,158,159,160,161,163,163,164,165,>
list_ironbow_g = [0,0,0,0,0,0,0,0,0,1,2,3,4,3,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,2,2,2,2,3,3,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,22,2>
list_ironbow_r = [0,0,0,0,0,0,0,0,0,0,0,0,0,2,5,9,12,16,19,23,26,29,33,36,39,43,46,49,52,54,57,60,63,66,69,71,74,77,80,83,85,88,91,94,96,99,102,105,107,110,112,115,117,120,122,124,127,129,131,133,136,138>
lut_ironbow = np.zeros((256, 1, 3), dtype=np.uint8)
lut_ironbow[:,:,0] = np.array(list_ironbow_b).reshape(256,1)
lut_ironbow[:,:,1] = np.array(list_ironbow_g).reshape(256,1)
lut_ironbow[:,:,2] = np.array(list_ironbow_r).reshape(256,1)

list_rainbow2 = [ 1, 3, 74, 0, 3, 74, 0, 3, 75, 0, 3, 75, 0, 3, 76, 0, 3, 76, 0, 3, 77, 0, 3, 79, 0, 3, 82, 0, 5, 85, 0, 7, 88, 0, 10, 91, 0, 14, 94, 0, 19, 98, 0, 22, 100, 0, 25, 103, 0, 28, 106, 0, 32,>
lut_rainbow2 = np.zeros((256, 1, 3), dtype=np.uint8)
lut_rainbow2[:,:,0] = np.array(list_rainbow2[2::3]).reshape(256,1)
lut_rainbow2[:,:,1] = np.array(list_rainbow2[1::3]).reshape(256,1)
lut_rainbow2[:,:,2] = np.array(list_rainbow2[0::3]).reshape(256,1)

colormaps = {
    'autumn': cv.COLORMAP_AUTUMN,
    'bone': cv.COLORMAP_BONE,
    'jet': cv.COLORMAP_JET,
    'winter': cv.COLORMAP_WINTER,
    'rainbow': cv.COLORMAP_RAINBOW,
    'ocean': cv.COLORMAP_OCEAN,
    'summer': cv.COLORMAP_SUMMER,
    'spring': cv.COLORMAP_SPRING,
    'cool': cv.COLORMAP_COOL,
    'hsv': cv.COLORMAP_HSV,
    'pink': cv.COLORMAP_PINK,
    'hot': cv.COLORMAP_HOT,
    'parula': cv.COLORMAP_PARULA,
    'magma': cv.COLORMAP_MAGMA,
    'inferno': cv.COLORMAP_INFERNO,
    'plasma': cv.COLORMAP_PLASMA,
    'viridis': cv.COLORMAP_VIRIDIS,
    'cividis': cv.COLORMAP_CIVIDIS,
    'twilight': cv.COLORMAP_TWILIGHT,
    'twilight_shifted': cv.COLORMAP_TWILIGHT_SHIFTED,
    'turbo': cv.COLORMAP_TURBO,
    'rainbow2': lut_rainbow2,
    'ironbow': lut_ironbow[-256:],
}

# Function to handle program exit
def signal_handler(sig, frame):
    mi48.stop()
    cv.destroyAllWindows()
    sys.exit(0)
def get_colormap(colormap='rainbow2', nc=None):
    """
    Return a 256-color LUT corresponding to `colormap`.

    `colormap` is either from open cv, matplotlib or explicitly defined above.
    If `nc` is not None, return a quantized colormap with `nc` different colors.
    """
    try:
        # use defualt opencv maps or explicitly defined above
        cmap = colormaps[colormap]
    except KeyError:
        cmap = cmapy.cmap(colormap)
    if nc is not None:
        # some names appear in both OpenCV (int), and Matplotlib (LUT)
        # attempt to pick up the one from Matplotlib
        if isinstance(cmap, int):
            try:
                cmap = cmapy.cmap(colormap)
            except KeyError:
                # return non-quantized CV cmap
                return cmap
        # we need to create a LUT with 256 entries, and these entries
        # are indexes in the actual color map; there are `nc` such indexes
        nmax = 256
        # number of indexes per color
        ipc = nmax // nc
        # If nmax is not multiple of nc, then we have to patch up the LUT.
        # Below, we choose to patch up with the highest index
        delta = nmax % nc
        lut = [int((j // ipc) / (nc-1) * (nmax-1)) for j in range(nmax-delta)]
        lut += [nmax-1,] * delta
        cmap = np.array([cmap[i] for i in lut], dtype='uint8')
    return cmap

# Set signal handlers for clean exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Connect to the MI48 thermal camera
mi48, connected_port, port_names = connect_senxor()

# Set desired FPS
STREAM_FPS = int(sys.argv[1]) if len(sys.argv) == 2 else 15
mi48.set_fps(STREAM_FPS)

# Configure MI48 filters
mi48.disable_filter(f1=True, f2=True, f3=True)
mi48.set_filter_1(85)
mi48.enable_filter(f1=True, f2=False, f3=False, f3_ks_5=False)
mi48.set_offset_corr(0.0)
mi48.set_sens_factor(100)

# Start thermal image capture
mi48.start(stream=True, with_header=True)

# Image processing parameters
par = {'blur_ks':3, 'd':5, 'sigmaColor': 27, 'sigmaSpace': 27}
dminav = RollingAverageFilter(N=10)
dmaxav = RollingAverageFilter(N=10)

# Function to encode image in base64
def encode_image_to_base64(image):
    _, buffer = cv.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Function to send data to the server
def send_data(image_base64, max_temp):
    url = 'http://192.168.1.XXX:5000/upload_ir'  # Adjust the endpoint
    data = {
        'image': image_base64,
        'max_temperature': float(max_temp)
    }
    requests.post(url, json=data)

# Main loop
while True:
    data, header = mi48.read()
    if data is None:
        mi48.stop()
        sys.exit(1)

    min_temp = dminav(data.min())
    max_temp = dmaxav(data.max())

    frame = data_to_frame(data, (80,62), hflip=False)
    frame = np.clip(frame, min_temp, max_temp)
    filt_uint8 = cv_filter(remap(frame), par, use_median=True, use_bilat=True, use_nlm=False)
    # colormap may be either a colormap list or a string
    data = filt_uint8
    resize=(400,310)
    colormap='rainbow2'
    interpolation=cv.INTER_CUBIC
    cmap = get_colormap(colormap, None)
    cvcol = cv.applyColorMap(data, cmap)
    if isinstance(resize, tuple) or isinstance(resize, list):
        cvresize =  cv.resize(cvcol, dsize=resize,
                            interpolation=interpolation)
    else:
        cvresize =  cv.resize(cvcol, dsize=None, fx=resize, fy=resize,
                            interpolation=interpolation)

    # Encode and send image & max temperature
    image = cv.rotate(cvresize, cv.ROTATE_90_COUNTERCLOCKWISE)
    image_base64 = encode_image_to_base64(image)
    send_data(image_base64, float(max_temp))

# Stop and cleanup
mi48.stop()
cv.destroyAllWindows()



