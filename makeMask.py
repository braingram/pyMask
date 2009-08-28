#!/myPython/bin/pythonw
#
#  makeMask.py
#  
#
#  Created by brett graham on 8/18/09.
#  Copyright (c) 2009 __MyCompanyName__. All rights reserved.
#

from pylab import *

# ===================================
#
#            Main functions
#
# ===================================

def ftModulus(ft):
	return sqrt(ft.real ** 2 + ft.imag ** 2)

def ftPhase(ft):
	return arctan2(ft.imag, ft.real)

def makeMaskOneChannel(image,randPhase=None):
	# make random phase shifted image
	if (randPhase is None):
		randPhase = 2.0 * pi * 1j * rand(*image.shape)
	return ifft2( ftModulus(fft2(image)) * exp(randPhase) ).real

def normalizeMask(image,mask):
	mScale = min(mean(image)/(mean(mask)-mask.min()), (1-mean(image))/(mask.max()-mean(mask)))
	return mScale*mask+(mean(image)-mean(mask)*mScale)

def makeMaskOneChannelNormalized(image, randPhase=None):
	return normalizeMask(image,makeMaskOneChannel(image,randPhase))

# ===================================
#
#         End of main functions
#
# ===================================

def normalize01(arr):
	return (arr - arr.min())/ (arr.max() - arr.min())

def makeMaskGrayscale(image):
	#mask = makeMaskOneChannel(image)
	
	# normalize mask
	#mScale = min(mean(image)/(mean(mask)-mask.min()), (1-mean(image))/(mask.max()-mean(mask)))
	#return mScale*mask+(mean(image)-mean(mask)*mScale)
	return makeMaskOneChannelNormalized(image)

def makeMaskRGB(image,perChannel=False):
	randPhase = 2.0 * pi * 1j * rand(*image.shape[:2])
	if perChannel:
		rm,gm,bm = [makeMaskOneChannelNormalized(image[:,:,i]) for i in range(3)]
	else:
		randPhase = 2.0 * pi * 1j * rand(*image.shape[:2])	
		rm,gm,bm = [makeMaskOneChannelNormalized(image[:,:,i], randPhase) for i in range(3)]
	# rm,gm,bm = [makeMaskGrayscale(image[:,:,i]) for i in range(3)]
	# rm,gm,bm = [normalizeMask(image[:,:,i],ifft2( ftModulus(fft2(image[:,:,i])) * exp(randPhase) ).real) for i in range(3)]
	# process each channel seperately
	#rm,gm,bm = [makeMaskOneChannelNormalized(image[:,:,i]) for i in range(3)]
	return dstack((rm,gm,bm))

def makeMask(image):
	# check if image is color
	if (len(image.shape) > 2):
		return makeMaskRGB(image)
	else:
		return makeMaskGrayscale(image)

def plotFtModulus(image,ft=None):
	if len(image.shape) > 2:
		r = ftModulus(fftshift(fft2(image[:,:,0])))
		g = ftModulus(fftshift(fft2(image[:,:,1])))
		b = ftModulus(fftshift(fft2(image[:,:,2])))
		imshow(normalize01(dstack((r,g,b))))
		#imshow(r)
		#figimage(normalize01(dstack((r,g,b))))
	else:
		if ft == None:
			ft = fft2(image)
		imshow(ftModulus(fftshift(ft)))

def plotFtPhase(image,ft=None):
	if len(image.shape) > 2:
		r = ftPhase(fftshift(fft2(image[:,:,0])))
		g = ftPhase(fftshift(fft2(image[:,:,1])))
		b = ftPhase(fftshift(fft2(image[:,:,2])))
		imshow(normalize01(dstack((r,g,b))))
		#imshow(r)
		#figimage(dstack((r,g,b)))
	else:
		if ft == None:
			ft = fft2(image)
		imshow(ftPhase(fftshift(ft)))

def testColor(imageFile,plotStats=False,perChannel=False):
	im = imread(imageFile)
	print im.shape
	# convert to float, no silly 0 to 255
	if im.dtype == uint8:
		im = im.astype(float)/255.0
	mask = makeMaskRGB(im,perChannel=perChannel)
	
	subplot(231); imshow(im)
	subplot(232); plotFtModulus(im)
	subplot(233); plotFtPhase(im)
	
	subplot(234); imshow(mask)
	subplot(235); plotFtModulus(mask)
	subplot(236); plotFtPhase(mask)

def testGray(imageFile,plotStats=False):
	gray()
	im = mean(imread(imageFile)[:,:,:3],2).astype(float)
	mask = makeMask(im)
	figure(1)
	imageFt = fftshift(fft2(im))
	subplot(241); imshow(im); title('Image');
	imageFtModulus = ftModulus(imageFt)
	def mid(arr):
		if mod(len(arr),2) == 1:
			return len(arr)/2
		else:
			return (len(arr)+1)/2
	subplot(242); imshow(log(imageFtModulus)); title('Log(Modulus)')
	imageFtModulus[mid(imageFtModulus[:,0]),mid(imageFtModulus[0])] = 0.0
	subplot(243); imshow(imageFtModulus); title('Log')
	subplot(244); imshow(ftPhase(imageFt)); title('Phase')
	
	maskFt = fftshift(fft2(mask))
	subplot(245); imshow(mask); title('Mask')
	maskFtModulus = ftModulus(maskFt)
	subplot(246); imshow(log(maskFtModulus)); title('Log(Modulus)')
	maskFtModulus[mid(maskFtModulus[:,0]),mid(maskFtModulus[0])] = 0.0
	subplot(247); imshow(maskFtModulus); title('Modulus')
	subplot(248); imshow(ftPhase(maskFt)); title('Phase')
	
	if plotStats or True:
		figure(2)
		subplot(211); hist(im.flatten()); title("Image")
		xlims = xlim(); ylims = ylim()
		subplot(212); hist(mask.flatten()); title("Mask")
		xlim(xlims); ylim(ylims)
		print "Image Mean:", mean(im)
		print "Mask  Mean:", mean(mask)

if __name__ == '__main__':
	import sys
	if (len(sys.argv) == 2):
		testGray(sys.argv[1])
	else:
		testGray('cln1.gif')
		#test('stp1.gif')
		#test('stp2.gif')
	show()