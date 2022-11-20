import os, cv2, time, datetime, pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import Sequence
from Helpers import *

INPUT_SHAPE = (256, 256, 3)
EPOCHS = 64
BATCH_SIZE = 1
TRAIN_PATH = os.path.join(os.getcwd(), f"{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} Dataset", "Training")
TRAIN_IDS = os.listdir(TRAIN_PATH)
VALIDATION_PATH = os.path.join(os.getcwd(), f"{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} Dataset", "Validation")
VALIDATION_IDS = os.listdir(VALIDATION_PATH)

trainDG = TilesMasksDataGenerator(TRAIN_IDS, TRAIN_PATH, imageSize=INPUT_SHAPE[0], batchSize=BATCH_SIZE)
trainSteps = len(TRAIN_IDS) // BATCH_SIZE

validationDG = TilesMasksDataGenerator(VALIDATION_IDS, VALIDATION_PATH, imageSize=INPUT_SHAPE[0], batchSize=BATCH_SIZE)
validationSteps = len(VALIDATION_IDS) // BATCH_SIZE

# Testing the data generator.
print(np.shape(trainDG.__getitem__(0)[0]))

# Create the model.
model = UNetModel(INPUT_SHAPE)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", DiceCoeff, IOUCoeff])
model.summary()

modelPath = f"Results/Model_BCE_Adam_{BATCH_SIZE}_{EPOCHS}_{INPUT_SHAPE[0]}_{INPUT_SHAPE[1]}.h5"

history = model.fit(
  trainDG,
  steps_per_epoch=trainSteps,
  validation_data=validationDG,
  validation_steps=validationSteps,
  epochs=EPOCHS,
  verbose=2,
  callbacks=[
    ModelCheckpoint(modelPath, save_best_only=True, verbose=1, monitor="val_loss", save_weights_only=False),
  ],
)

StorePickle(history.history, modelPath.replace(".h5", ".pkl").replace("Model", "History"))
