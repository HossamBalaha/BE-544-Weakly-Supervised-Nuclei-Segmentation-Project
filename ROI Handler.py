import os

# # Add the OpenSlide DLL.
os.add_dll_directory(r"C:\Program Files\openslide-win64-20221111\bin")

from Helpers import *

tileWidth = 256
tileHeight = 256
imageLevel = 0

projectPath = os.path.join(os.getcwd(), "QuPath Project", 'project.qpproj')
imagePath = os.path.join(os.getcwd(), "Slides", "TCGA-AA-3562-01A-01-BS1.2bc37a71-647d-4e7e-9e3a-12191942a051.svs")

storePath = os.path.join(os.getcwd(), f"{tileWidth}x{tileHeight} Dataset")
if not os.path.exists(storePath):
  os.mkdir(storePath)
  os.mkdir(os.path.join(storePath, "Training"))
  os.mkdir(os.path.join(storePath, "Testing"))

print("Project Path:", projectPath)
print("Store Path:", storePath)
print("Image Path:", imagePath)

# Part B
# Part A
# ROI-B1
# Annotation-B1
# ROI-A1
# Annotation-A1

# Handle the ROI-A1 + Annotation-A1.
HandleROI(projectPath, imagePath, 4, 5, os.path.join(storePath, "Training"), imageLevel, tileWidth, tileHeight)

# Handle the ROI-B1 + Annotation-B1.
HandleROI(projectPath, imagePath, 2, 3, os.path.join(storePath, "Testing"), imageLevel, tileWidth, tileHeight)
