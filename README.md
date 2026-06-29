# MY_PROJECTS

# Olivetti Faces Classification using CNN 

This repository contains a Deep Learning project for classifying face images from the *Olivetti Faces dataset* using a (CNN) built with TensorFlow and Keras.

---

## Technologies Used
* **Python**
* **TensorFlow / Keras** (For building and training the CNN model)
* **OpenCV** (For image resizing and preprocessing)
* **Scikit-learn** (For fetching the dataset and splitting it into Train & Test sets)
* **Matplotlib** (For data visualization and displaying results)

---

##  Project Workflow & Pipeline

1. **Dataset Loading:** 
   Utilized the `fetch_olivetti_faces` dataset from `sklearn`, which contains 400 grayscale images of 40 distinct subjects.

2. **Image Preprocessing:**
   * Resized all images from their original dimensions to 48x48 pixels using OpenCV (`cv2.resize`).
   * Reshaped the image matrices into (-1, 48, 48, 1) to fit the channel requirements of a CNN model.

3. **Data Splitting:**
   Split the dataset into **80% Training** and **20% Testing** using `train_test_split` with stratification to ensure a balanced representation of all classes.

4. **Data Augmentation (Resource-Constrained Optimization):**
   Due to the limited size of the dataset and available training images, data augmentation layers were integrated directly into the model architecture. This approach synthetically expands the dataset, prevents overfitting, and improves generalization without requiring additional data collection:
   * `RandomFlip("horizontal")`
   * `RandomRotation(0.1)`

5. **Model Architecture & Design Rationale (CNN):**
   * **Input Layer:** Accepts $(48, 48, 1)$ images, matching the preprocessed grayscale face matrices.
   * **`Conv2D` Layers:** Two layers ($64$ and $128$ filters) with `relu` activation were utilized to automatically extract hierarchical spatial features (like edges, shapes, and facial components) from the images.
   * **`MaxPooling2D` Layers:** Applied after each convolutional layer to downsample the feature maps. This reduces computational complexity, minimizes the number of parameters, and helps achieve translation invariance.
   * **`Flatten` Layer:** Converts the 2D feature maps into a 1D vector so they can be fed into the dense layers.
   * **`Dense` (Fully Connected) Layers:** A hidden layer with $128$ units and `relu` activation to learn non-linear combinations of the extracted features, followed by a final `Dense` layer with `softmax` activation to output a probability distribution across all 40 distinct subject classes.

6. **Training:**
   Compiled the model using the `adam` optimizer and `sparse_categorical_crossentropy` loss, then trained it for **50 epochs**.

---

##  Inference & Results

The final block of the notebook performs real-time inference on a test image, outputting the actual class, predicted class, and the model's confidence percentage.
