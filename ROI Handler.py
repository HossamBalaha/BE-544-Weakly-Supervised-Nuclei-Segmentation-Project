import os

# # Add the OpenSlide DLL.
os.add_dll_directory(r"C:\Program Files\openslide-win64-20221111\bin")

import cv2, openslide
import numpy as np
from paquo.projects import QuPathProject

tileWidth = 50
tileHeight = 50
imageLevel = 0

projectPath = os.path.join(os.getcwd(), "QuPath Project", 'project.qpproj')
imagePath = os.path.join(os.getcwd(), "TCGA-AA-3562-01A-01-BS1.2bc37a71-647d-4e7e-9e3a-12191942a051.svs")

storePath = os.path.join(os.getcwd(), f"{tileWidth}x{tileHeight} Dataset")
if not os.path.exists(storePath):
  os.mkdir(storePath)
  os.mkdir(os.path.join(storePath, "Training"))
  os.mkdir(os.path.join(storePath, "Testing"))

print("Project Path:", projectPath)
print("Store Path:", storePath)
print("Image Path:", imagePath)


def HandleROI(projectPath, imagePath, roiIndex, annotationIndex, saveFolder):
  # Read the QuPath project.
  project = QuPathProject(projectPath, mode='r')

  # Get the image.
  image = project.images[0]
  print("Image:", image)

  # Get the annotations.
  annotations = image.hierarchy.annotations
  # for annotation in annotations:
  #   print(annotation.name, annotation.path_class)

  # Get the annotations.
  roi = annotations[roiIndex]  # ROI-A1
  annotation = annotations[annotationIndex]

  # Get the dimensions of the working ROIs.
  xy = roi.roi.exterior.xy
  xLeft = int(xy[0][0])
  xRight = int(xy[0][2])
  yTop = int(xy[1][0])
  yBottom = int(xy[1][1])

  print("ROI Dimensions:", xLeft, xRight, yTop, yBottom)

  # Get the ROIs.
  roiAnnotation = annotation.roi

  # Get the coords.
  xList = [np.array(el.exterior.xy[0]) for el in roiAnnotation.geoms]
  yList = [np.array(el.exterior.xy[1]) for el in roiAnnotation.geoms]

  s = openslide.OpenSlide(imagePath)
  print("Image Dimensions:", s.dimensions)

  # Get the image dimensions.
  imageWidth = s.dimensions[0]
  imageHeight = s.dimensions[1]

  # Get the image downsample.
  imageDownsample = int(s.level_downsamples[imageLevel])

  # Get the image size.
  imageSize = s.level_dimensions[imageLevel]

  # Read the image.
  image = s.read_region((0, 0), imageLevel, imageSize)

  # Downsample the x and y coords.
  xList = [x_ / imageDownsample for x_ in xList]
  yList = [y_ / imageDownsample for y_ in yList]

  polygons = [
    np.array([(x_, y_) for x_, y_ in zip(xList[i], yList[i])]).astype(np.int32)
    for i in range(len(xList))
  ]
  # Cast the polygons.
  polygons = np.array(polygons, dtype=object)

  print("Number of Polygons:", len(polygons))

  # Downsample the dimensions of the ROIs.
  xLeft = int(xLeft / imageDownsample)
  xRight = int(xRight / imageDownsample)
  yTop = int(yTop / imageDownsample)
  yBottom = int(yBottom / imageDownsample)

  # Convert the image to a NumPy array.
  imageLR = np.array(image).astype(np.uint8)

  # Convert the image to BGR.
  imageLR = cv2.cvtColor(imageLR, cv2.COLOR_RGB2BGR)

  # Create a mask.
  mask = np.zeros(imageLR.shape, dtype=np.uint8)

  # Fill the mask with polygons.
  cv2.fillPoly(mask, polygons, (255, 255, 255))

  # Draw the polygons.
  # imageLRContours = cv2.drawContours(imageLR, polygons, -1, (0, 255, 0), 1)

  croppedImageLR = imageLR[yTop:yBottom, xLeft:xRight, :]
  croppedMaskLR = mask[yTop:yBottom, xLeft:xRight, :]

  print("Cropped Image Dimensions:", croppedImageLR.shape)
  print("Cropped Mask Dimensions:", croppedMaskLR.shape)

  # Display the image.
  # cv2.imshow("ROI LR Image", croppedImageLR)
  # cv2.waitKey(0)

  # Get the number of tiles.
  numXTiles = int((xRight - xLeft) / tileWidth)
  numYTiles = int((yBottom - yTop) / tileHeight)

  print("Number of Tiles:", numXTiles, numYTiles)

  # Extract the tiles.
  for i in range(numXTiles):
    for j in range(numYTiles):
      xLeft_ = i * tileWidth
      xRight_ = xLeft_ + tileWidth
      yTop_ = j * tileHeight
      yBottom_ = yTop_ + tileHeight
      tile_ = croppedImageLR[yTop_:yBottom_, xLeft_:xRight_, :]
      mask_ = croppedMaskLR[yTop_:yBottom_, xLeft_:xRight_, :]
      conc_ = cv2.hconcat([tile_, mask_])
      cv2.imwrite(os.path.join(saveFolder, "Tile_Mask_" + str(i + 1) + ".png"), conc_)


# Part B None
# Part A None
# ROI-A1 None
# Annotation-A1 QuPathPathClass('Nuclei')
# ROI-B1 None
# Annotation-B1 QuPathPathClass('Nuclei')

# Handle the ROI-A1.
HandleROI(projectPath, imagePath, 2, 3, os.path.join(storePath, "Training"))

# Handle the ROI-B1.
HandleROI(projectPath, imagePath, 4, 5, os.path.join(storePath, "Testing"))
