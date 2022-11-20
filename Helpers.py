import os, cv2, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence


class TilesMasksDataGenerator(Sequence):
  def __init__(self, ids, path, batchSize=8, imageSize=100):
    self.ids = np.random.permutation(ids)
    self.path = path
    self.batchSize = batchSize
    self.imageSize = imageSize
    self.on_epoch_end()

  def __load__(self, idName):
    imagePath = os.path.join(self.path, idName)
    image = cv2.imread(imagePath)
    w_ = int(image.shape[1] / 2)
    mask_ = image[:, w_:, :]
    im_ = image[:, :w_, :]
    mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
    im_ = im_ / 255.0
    mask_ = mask_ / 255.0
    return im_, mask_

  def __getitem__(self, index):
    if ((index + 1) * self.batchSize > len(self.ids)):
      self.batchSize = len(self.ids) - index * self.batchSize
    filesBatch = self.ids[index * self.batchSize: (index + 1) * self.batchSize]
    images, masks = [], []
    for idName in filesBatch:
      img_, mask_ = self.__load__(idName)
      images.append(img_)
      masks.append(mask_)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

  def on_epoch_end(self):
    self.ids = np.random.permutation(self.ids)

  def __len__(self):
    return int(np.ceil(len(self.ids) / float(self.batchSize)))


def HandleROI(
    projectPath, imagePath, roiIndex, annotationIndex, saveFolder, imageLevel=0, tileWidth=256,
    tileHeight=256
):
  import openslide
  from paquo.projects import QuPathProject

  # Read the QuPath project.
  project = QuPathProject(projectPath, mode='r')

  # Get the image.
  image = project.images[0]
  print("Image:", image)

  # Get the annotations.
  annotations = image.hierarchy.annotations
  for annotation in annotations:
    print(annotation.name)  # , annotation.path_class

  # Get the annotations.
  roi = annotations[roiIndex]
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
  imageWidth, imageHeight = s.dimensions[0], s.dimensions[1]

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
      cv2.imwrite(os.path.join(saveFolder, f"Tile_Mask_{i + 1}_{j + 1}.png"), conc_)


def UNetModel(inputShape):
  from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, Concatenate
  from tensorflow.keras.models import Model
  from tensorflow.keras.backend import clear_session

  def _downBlock(x, filters, kernelSize=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

  def _upBlock(x, skip, filters, kernelSize=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(c)
    return c

  def _bottleneck(x, filters, kernelSize=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation="relu")(c)
    return c

  clear_session()
  f = [16, 16, 32, 64, 128, 256]
  inputs = Input(inputShape)

  p0 = inputs
  c1, p1 = _downBlock(p0, f[0])  # 128 -> 64
  c2, p2 = _downBlock(p1, f[1])  # 64 -> 32
  c3, p3 = _downBlock(p2, f[2])  # 32 -> 16
  c4, p4 = _downBlock(p3, f[3])  # 16->8
  c5, p5 = _downBlock(p4, f[4])  # 16->8

  bn = _bottleneck(p5, f[5])

  u0 = _upBlock(bn, c5, f[4])  # 8 -> 16
  u1 = _upBlock(u0, c4, f[3])  # 8 -> 16
  u2 = _upBlock(u1, c3, f[2])  # 16 -> 32
  u3 = _upBlock(u2, c2, f[1])  # 32 -> 64
  u4 = _upBlock(u3, c1, f[0])  # 64 -> 128

  outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
  model = Model(inputs, outputs)
  return model


# https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
def DiceCoeff(yTrue, yPred, smooth=1):
  import tensorflow as tf

  yTrue_ = tf.reshape(yTrue, [-1])
  yPred_ = tf.reshape(yPred, [-1])
  intersection = tf.reduce_sum(yTrue_ * yPred_)
  score = (2.0 * intersection + smooth) / (tf.reduce_sum(yTrue_) + tf.reduce_sum(yPred_) + smooth)
  return score


def DiceLoss(yTrue, yPred):
  loss = 1 - DiceCoeff(yTrue, yPred)
  return loss


def BCEDiceLoss(yTrue, yPred):
  from tensorflow.keras.losses import binary_crossentropy

  loss = binary_crossentropy(yTrue, yPred) + DiceCoeff(yTrue, yPred)
  return loss


def IOUCoeff(yTrue, yPred, smooth=1):
  import tensorflow as tf

  intersection = tf.reduce_sum(yTrue * yPred)
  union = tf.reduce_sum(yTrue) + tf.reduce_sum(yPred) - intersection
  iou = (intersection + smooth) / (union + smooth)
  return iou


def IOULoss(yTrue, yPred):
  loss = 1 - IOUCoeff(yTrue, yPred)
  return loss


def BCEIOULoss(yTrue, yPred):
  from tensorflow.keras.losses import binary_crossentropy

  loss = binary_crossentropy(yTrue, yPred) + IOULoss(yTrue, yPred)
  return loss


def StorePickle(data, path):
  with open(path, 'wb') as f:
    pickle.dump(data, f)


def LoadPickle(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data
