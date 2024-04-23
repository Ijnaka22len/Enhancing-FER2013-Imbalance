# %% [markdown]
# # Enhancing Facial Expression Recognition Using Class Imbalance Handling Techniques.
# 
# 
# 
# 
# <br><br><br><br><br>
# **Leonel Akanji Akanji &nbsp;&nbsp;&nbsp; 0986513**<br>
# **Mst Rumana Sumi &nbsp;&nbsp;&nbsp;   0969068**<br>

# %%
from google.colab import drive
drive.mount('/content/drive')
working_dir =  '/content/drive/MyDrive/CU/Spring2024/AppliedML_IA651/Final_Project/' # If we have to both edit using this same notebook, you can download these files to your respective drives and connect to the dataset from your google drive.


# %% [markdown]
# # 1) Importing Required Libraries

# %%
import pickle # Used to save history of training progress
import random
import datetime # Getting timestamp: Used to get date for labelling model using the function ModelCheckpoint
import numpy as np # used for most mathematical operations
import pandas as pd # Used to create a dataframe from detaset for the model
import  matplotlib.pyplot as plt # Use for visualization by plotting data or data points
from keras.models import Sequential
from keras.models import load_model # Used to load pretrained model for testing
from keras.utils import to_categorical # used to label the data
from keras.callbacks import ModelCheckpoint # Used to save best model weights during training process
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam # Model optimizer: Learning rate of 1e-5 is used
from sklearn.model_selection import train_test_split # Used to split data into Training, Validation & Testing
from keras.preprocessing.image import ImageDataGenerator # Used for data augmentation: mainly to reduce within class imbalance to improve accurracy of the model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # Early stopping helps to stop training when model is no longer learning
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation # Functions used to build the CNN architecture of the model

# %% [markdown]
# # 2) Custom Functions

# %%
def getDataCount(emotion_map=None, data=None, ):
  emotion_counts = data['emotion'].value_counts(sort=False).reset_index()
  emotion_counts.columns = ['emotion', 'number']
  emotion_counts['emotion'] = emotion_counts['emotion'].map(emotion_map)
  return emotion_counts

#*******************************************************************************
def barplot(data=None, emotion_map=None, plt=None, titleType='Emotions'):
    emotion_counts = getDataCount(emotion_map, data=data)
    # print(emotion_counts)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    plt.figure(figsize=(8, 6))
    bars = plt.bar(emotion_counts['emotion'], emotion_counts['number'], color=colors)
    plt.title('Class Distribution Of '+ titleType)
    plt.xlabel('Emotions')
    plt.ylabel('Number Of Samples')
    plt.xticks(rotation=45)

    for bar, count in zip(bars, emotion_counts['number']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), count,
                ha='center', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.show()

#*******************************************************************************
def plotDatasetSamples(file_path=None, emotion_map=None ,max_num_plot=10):
    if max_num_plot % 3 != 0:
        print(f"'max_num_plot' cannot be odd. Defaulting to max_num_plot = {max_num_plot - 1}")
        max_num_plot = max_num_plot - 1

    df = pd.read_csv(file_path)
    df['pixels'] = df['pixels'].apply(lambda x: [int(p) for p in x.split()])
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x).reshape(48, 48, 1))
    samples = df.sample(max_num_plot)
    samples['emotion_label'] = samples['emotion'].map(emotion_map)

    # Plotting the pixel data
    plt.figure(figsize=(12, 6))
    for i in range(max_num_plot):
        plt.subplot(3, 5, i + 1)
        plt.imshow(samples['pixels'].values[i].reshape(48, 48), cmap='gray')
        plt.title(f"{samples['emotion_label'].values[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#*******************************************************************************
def downSampleData(data=None, emotion_map=None):
    # print('----------------------------------------------\n',data)
    min_count = getDataCount(emotion_map, data=data)['number'].min()
    grouped = data.groupby('emotion')
    down_sampled = grouped.apply(lambda x: x.sample(min_count))
    down_sampled.reset_index(drop=True, inplace=True)
    # print(down_sampled)
    return down_sampled

#*******************************************************************************
from collections import defaultdict

def preProcess_augment_oversample(file_path=None, datagen=None, target_samples=None, imbalance_threshold=None):
    # Part 1: Pre-processing
    df = pd.read_csv(file_path)
    df['pixels'] = df['pixels'].apply(lambda x: [int(p) for p in x.split()])  # Convert strings to lists of integers
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x).reshape(48, 48, 1))  # Reshape to 48x48x1
    df['pixels'] = df['pixels'] / 255.0  # Normalisation

    data = np.array(df['pixels'].tolist())
    labels = np.array(df['emotion'])

    if imbalance_threshold:
          # Calculate class counts
          class_counts = defaultdict(list)
          for idx, label in enumerate(labels):
              class_counts[label].append(idx)

          # Determine classes below and above threshold
          minority_classes = [label for label, idxs in class_counts.items() if len(idxs) < imbalance_threshold]
          majority_classes = [label for label, idxs in class_counts.items() if len(idxs) >= imbalance_threshold]

          # Oversampling minority classes below threshold
          oversampled_data = []
          oversampled_labels = []
          for label in minority_classes:
              indices = class_counts[label]
              num_to_generate = imbalance_threshold - len(indices)
              if num_to_generate > 0:
                  for idx in np.random.choice(indices, num_to_generate, replace=True):
                      augmented_x = datagen.random_transform(data[idx])
                      oversampled_data.append(augmented_x)
                      oversampled_labels.append(labels[idx])

          # Downsampling majority classes above threshold
          downsampled_data = []
          downsampled_labels = []
          for label in majority_classes:
              indices = class_counts[label]
              num_to_keep = imbalance_threshold
              if num_to_keep > 0:
                  sampled_indices = np.random.choice(indices, num_to_keep, replace=False)
                  for idx in sampled_indices:
                      downsampled_data.append(data[idx])
                      downsampled_labels.append(labels[idx])

          # Combine oversampled and downsampled data
          if oversampled_data:
              data = np.concatenate((data, oversampled_data), axis=0)
              labels = np.concatenate((labels, oversampled_labels), axis=0)

          if downsampled_data:
              data = np.concatenate((data, downsampled_data), axis=0)
              labels = np.concatenate((labels, downsampled_labels), axis=0)

    # Part 2: Augmentation
    expanded_data = []
    expanded_labels = []
    if target_samples is None:
      expanded_data=data
      expanded_labels=labels
    else:
      for x, y in zip(data, labels):
          expanded_data.append(x)
          expanded_labels.append(y)

          # Augmentation and adding the augmented image
          if target_samples:
              for i in range(target_samples):
                  augmented_x = datagen.random_transform(x)
                  expanded_data.append(augmented_x)
                  expanded_labels.append(y)

    expanded_data = np.array(expanded_data)
    expanded_labels = np.array(expanded_labels)

    # Part 3: Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(expanded_data, expanded_labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Part 4: Convert integer labels to one-hot encoded format
    y_train = to_categorical(y_train, num_classes=7)  # for 7 classes
    y_val = to_categorical(y_val, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

#*******************************************************************************
def create_CNN_model(input_shape=None):
    if input_shape is None :
        input_shape=(48,48,1)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, Activation('softmax')))

    return model

#*******************************************************************************
def plotModelHistory(history=None, plt=None):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # history for accuracy
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'validate'], loc='upper left')
    axs[0].grid(True)
    # history for loss
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    axs[1].grid(True)
    plt.show()

#*******************************************************************************
def plotLoadedModelHistory(history_dict=None, plt=None):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # history for accuracy
    axs[0].plot(history_dict['accuracy'])
    axs[0].plot(history_dict['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')
    axs[0].grid(True)

    # history for loss
    axs[1].plot(history_dict['loss'])
    axs[1].plot(history_dict['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')
    axs[1].grid(True)
    plt.show()

#*********************************************************************************
def plotConfusionMatrix(model=None, X_test=None, y_test=None, emotion_map=None, plt=None, np=None):
  # predictions of model
  y_pred = model.predict(X_test)
  y_pred_classes = np.argmax(y_pred, axis=1)

  # Compute confusion matrix
  conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)

  # Plot confusion matrix using Matplotlib
  plt.figure(figsize=(8, 6))
  plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
  plt.colorbar()
  # Set tick marks and labels
  tick_marks = np.arange(len(emotion_map))
  plt.xticks(tick_marks, emotion_map.values(), rotation=45)
  plt.yticks(tick_marks, emotion_map.values())

  for i in range(len(emotion_map)):
      for j in range(len(emotion_map)):
          plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')

  plt.tight_layout()
  plt.show()

#*******************************************************************************
def plotPredictions(X_test=None, y_test=None,  model=None, plt=None, emotion_map=None, max_num_plot=10 ):
  if max_num_plot % 3 != 0:
    print(f"'max_num_plot' cannot be old. Defaulting to max_num_plot= {max_num_plot} -1 = {max_num_plot-1}")
    max_num_plot = max_num_plot- 1

  predictions = model.predict(X_test)
  predicted_labels = np.argmax(predictions, axis=1)

  # Calculate accuracy
  correct_predictions = np.equal(predicted_labels, np.argmax(y_test, axis=1))
  accuracy = np.mean(correct_predictions)
  print(f"Test Accuracy: {accuracy * 100:.2f}%")

  # Create a list of emotions from the predicted labels using the emotion_map
  predicted_emotions = [emotion_map[label] for label in predicted_labels]
  actual_emotions = [emotion_map[label] for label in np.argmax(y_test, axis=1)]


  plt.figure(figsize=(12, 6))
  for i in range(1,max_num_plot+1):
      plt.subplot(3, max_num_plot // 3, i)
      plt.imshow(X_test[i].reshape(48, 48), cmap='gray')
      plt.title(f"Actual: {actual_emotions[i]}\nPredicted: {predicted_emotions[i]}",fontsize=8)
      plt.axis('off')

  plt.tight_layout()
  plt.show()

#*******************************************************************************
def sampleAugmentationPlotOnly(X_train=None):
  rotate_datagen = ImageDataGenerator(vertical_flip=True, fill_mode='nearest')  # Rotate images by up to angle = 40 degrees
  horizontal_flip = ImageDataGenerator(horizontal_flip=True,fill_mode='nearest')  # Flip images horizontally
  datagen = [rotate_datagen, horizontal_flip]
  aug_titles = ["Vertical flip Augmentation", "Horizontal flip Augmentation"]

  # Create a figure with two rows and two columns of subplots
  fig, axs = plt.subplots(2, 2, figsize=(10, 10))
  index = random.randint(0, len(X_train) - 1)  # Choose a random image index
  for i in range(2):
      augmented_image = datagen[i].random_transform(X_train[index])  # Augment a random image
      # Plot the original image
      axs[i, 0].imshow(X_train[index], cmap='gray')  # Force grayscale interpretation
      axs[i, 0].set_title("Original Image")
      axs[i, 0].axis('off')  # Remove axis numbering (ticks)
      # Plot the augmented image
      axs[i, 1].imshow(augmented_image, cmap='gray')  # Force grayscale interpretation
      axs[i, 1].set_title(aug_titles[i])
      axs[i, 1].axis('off')  # Remove axis numbering (ticks)
  plt.show()

# %% [markdown]
# # 3) Dataset Overview

# %% [markdown]
# Lets import the data as a dataframe and visualise its content

# %%
df = pd.read_csv(working_dir+'fer2013.csv')
print(df.head())
print(df.describe())


# %% [markdown]
# Graphical visualisation of data &  target labels

# %%
emotion_map = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
barplot(data=df,
        emotion_map=emotion_map,
        plt=plt,
        titleType='Emotions(Raw Data)')

del df

# %% [markdown]
# Let's Plot samples of the pixels images with labels

# %%
plotDatasetSamples(file_path=working_dir+'fer2013.csv',
                   emotion_map=emotion_map,
                   max_num_plot=15
                   )


# %% [markdown]
# # 4) Data Pre-processing
# **Steps:**
# - Splitting dataset into 3 parts: Training: **80%**, Validation: **10% **& Testing: **10%** :Part 3 of  `preProcess_augment_oversample()`
# - Convert strings to lists of integers: Part 1 of  `preProcess_augment_oversample()`
# - Reshape to 48x48x1 and normalise grayscale images: Part 1 of   `preProcess_augment_oversample()`
# -  Perform one-hot encoding label, e.g. class 3 to [0,0,0,1,0,0,0]: Part 4 of `preProcess_augment_oversample()`
# - Oversampling minority classes : `preProcess_augment_oversample()`
# - Data augmentation: `preProcess_augment_oversample()`

# %%
datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                            #  rotation_range=15,
                            #  width_shift_range=0.2,
                            #  height_shift_range=0.2,
                             fill_mode='nearest' # determines how empty pixels(if available) are filled based on the nearest available pixel values, during  image transformation
                             )
imbalance_threshold = 5121
(X_train, y_train), (X_val, y_val), (X_test, y_test) = preProcess_augment_oversample(
    file_path=working_dir+'fer2013.csv',
    datagen=datagen,
    target_samples=None,
    imbalance_threshold=imbalance_threshold
    )
X_train.shape, X_val.shape, X_test.shape

# %% [markdown]
# Sample Augmentation plot only

# %%
sampleAugmentationPlotOnly(X_train=X_train)

# %% [markdown]
# # 5) CNN Architecture
# 
#  - `create_CNN_model()`
# 
# 

# %%
model =create_CNN_model()
# model.summary()

# %% [markdown]
# # 6) Training

# %% [markdown]
# **Callbacks**
# - `ModelCheckpoint()`: Used to save best model weights during training process
# - `EarlyStopping()`:  Early stopping helps to stop training when model is no longer learning

# %%
current_time = datetime.datetime.now()
checkpoint_path = working_dir+f'checkpoint/best_model_{str(current_time).split(" ")[0]}.h5'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')  # 'max' if monitoring accuracy, 'min' if monitoring loss

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# %% [markdown]
# **Compile and Train Model**
# - epochs=60

# %%
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    batch_size=32,
                    epochs=60,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop, checkpoint],
                    verbose=1,

                    )

# %% [markdown]
# **Save the training history dictionary**

# %%
history_path = working_dir+f'checkpoint/best_model_history.pkl'

with open(history_path, 'wb') as file:
    pickle.dump(history.history, file)


# %% [markdown]
# **Visualize Training Performance**

# %%
plotModelHistory(history=history, plt=plt)

# %% [markdown]
# # 7) Testing trained cnn model

# %%
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# %% [markdown]
# **Confusion Matrix**

# %%
plotConfusionMatrix(model=model,
                    X_test=X_test,
                    y_test=y_test,
                    emotion_map=emotion_map,
                    plt=plt,
                    np=np
                    )

# %%
plotPredictions(X_test=X_test,
                y_test=y_test,
                model=model,
                plt=plt,
                emotion_map=emotion_map,
                max_num_plot = 15 )

# %% [markdown]
# # 8) Tesing on Loaded CNN Model

# %%
emotion_map = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
datagen = ImageDataGenerator(horizontal_flip=True,
                             fill_mode='nearest' # determines how empty pixels(if available) are filled based on the nearest available pixel values, during  image transformation
                             )

# imbalance_threshold = 547
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = preProcess_augment_oversample(
#     file_path=working_dir+'fer2013.csv',
#     datagen=datagen,
#     target_samples=None,
#     imbalance_threshold=imbalance_threshold
#     )
# X_train.shape, X_val.shape, X_test.shape

# %%
import os
os.listdir(working_dir+'checkpoint')

# %%
loaded_model = load_model(working_dir+'checkpoint/best_model_2024-04-23.h5', {'Activation': Activation} )

# %% [markdown]
# **Visualize Training Performance of loaded CNN model**

# %%
history_path= working_dir + "checkpoint/best_model_history.pkl"
with open(history_path, 'rb') as file:
    loaded_history = pickle.load(file)

plotLoadedModelHistory(history_dict=loaded_history, plt=plt)

# %%
loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss}")


# %% [markdown]
# **Confusion Matrix of Loaded Model**

# %%
plotConfusionMatrix(model=loaded_model,
                    X_test=X_test,
                    y_test=y_test,
                    emotion_map=emotion_map,
                    plt=plt,
                    np=np
                    )

# %%
plotPredictions(X_test=X_test,
                y_test=y_test,
                model=loaded_model,
                plt=plt,
                emotion_map=emotion_map,
                max_num_plot = 15 )

# %% [markdown]
# # END


