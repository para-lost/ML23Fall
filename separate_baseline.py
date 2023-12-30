
# 可以自己用少量数据集简单训练一个小区分模型做naive的baseline，但是只包含主要器官，像血管之类的不太行
# 一个改编的实例：最简单的分类器，但是这个是直接输入图片进行判断，没有分割能力，也就是说只能直接手动指定一块区域让分类
# (上个年代的东西)
!pip install kaggle
!kaggle datasets download -d mdwaquarazam/microorganism-image-classification

# get class names
import pathlib
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg

micro_organism_directory = "/content/gdrive/MyDrive/Micro-Organism/Micro_Organism"

data_dir = pathlib.Path(micro_organism_directory)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))


def view_random_image(target_dir, target_class):
  target_folder = target_dir + "/" + target_class
  random_image = random.sample(os.listdir(target_folder), 1) # get 1
  print(random_image)
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape: {img.shape}")
  return img

# img = view_random_image(target_dir = micro_organism_directory,
#                         target_class = random.choice(class_names))


# data process
train_datagen = ImageDataGenerator(rescale = 1/255.,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split = 0.2)

train_data = train_datagen.flow_from_directory(micro_organism_directory,
                                               target_size = (224, 224),
                                               batch_size = 32,
                                               class_mode = "categorical",
                                               subset = "training") # set as training data

test_data = train_datagen.flow_from_directory(micro_organism_directory,
                                               target_size = (224, 224),
                                               batch_size = 32,
                                               class_mode = "categorical",
                                               subset='validation') # set as validation data

# train一个小模型来对比
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model_1 = Sequential([
    Conv2D(10, 3, input_shape = (224, 224, 3), activation = "relu"),
    Conv2D(10, 3, activation = "relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation = "relu"),
    Conv2D(10, 3, activation = "relu"),
    MaxPool2D(),
    Flatten(),
    Dense(8, activation = "softmax")
])

model_1.compile(loss = "categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

history_1 = model_1.fit(train_data,
            epochs = 5,
            steps_per_epoch = len(train_data),
            validation_data = test_data,
            validation_steps = len(test_data))

# visualize
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"])) 

  plt.plot(epochs, loss, label = "Training_loss")
  plt.plot(epochs, val_loss, label = "Val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.figure() 
  plt.plot(epochs, accuracy, label = "Training_accuracy")
  plt.plot(epochs, val_accuracy, label = "Val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()


# plot_loss_curves(history_1)

def pred_and_plot(model, filename, class_names = class_names):

  def load_and_prep_image(filename, img_shape = 224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size = [img_shape, img_shape])
    img = img/255.
    return img

  img = load_and_prep_image(filename)

  pred = model.predict(tf.expand_dims(img, axis = 0))
  print(pred)

  if(len(pred[0]) > 1):
    pred_class = class_names[tf.argmax(pred[0])]
  else:
    pred_class = class_names[int(tf.round(pred))]

  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)

pred_and_plot(model = model_1,
              filename = "xxx.jpg",
              class_names = class_names)



# 目前主流还是各种魔改unet



