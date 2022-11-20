import os, pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Helpers import *

modelPath = f"Results/Model_BCE_Adam_1_64_256_256.h5"

history = LoadPickle(modelPath.replace(".h5", ".pkl").replace("Model", "History"))

# Plot the training and validation metrics.
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.legend()
plt.tight_layout()
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(history["DiceCoeff"], label="Training Dice")
plt.plot(history["val_DiceCoeff"], label="Validation Dice")
plt.legend()
plt.tight_layout()
plt.grid()
plt.subplot(2, 2, 4)
plt.plot(history["IOUCoeff"], label="Training IoU")
plt.plot(history["val_IOUCoeff"], label="Validation IoU")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(modelPath.replace(".h5", "_History.pdf"))

INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 1
TEST_PATH = os.path.join(os.getcwd(), f"{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]} Dataset", "Testing")
TEST_IDS = os.listdir(TEST_PATH)

testDG = TilesMasksDataGenerator(TEST_IDS, TEST_PATH, imageSize=INPUT_SHAPE[0], batchSize=BATCH_SIZE)


model = load_model(modelPath, custom_objects={"DiceCoeff": DiceCoeff, "IOUCoeff": IOUCoeff})

plt.figure(figsize=(30, 10))

for i in range(10):
  x, y = testDG.__getitem__(i)
  predictions = model.predict(x)

  im = np.array(predictions[0] * 255, dtype='uint8')
  im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
  gt = cv2.cvtColor(np.array(y[0, :, :] * 255, dtype='uint8'), cv2.COLOR_GRAY2BGR)
  inp = np.array(x[0, :, :] * 255, dtype='uint8')

  plt.subplot(3, 10, (i + 1))
  plt.imshow(inp)
  plt.axis('off')
  plt.tight_layout()
  plt.title("Input")
  plt.subplot(3, 10, (i + 1) + 10)
  plt.imshow(gt)
  plt.axis('off')
  plt.tight_layout()
  plt.title("Ground Truth")
  plt.subplot(3, 10, (i + 1) + 20)
  plt.title("Prediction")
  plt.imshow(im)
  plt.axis('off')
  plt.tight_layout()
plt.savefig(modelPath.replace(".h5", "_Testing.pdf"))
# plt.show()

