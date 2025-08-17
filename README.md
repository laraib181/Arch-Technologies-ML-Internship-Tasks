# 📌 Machine Learning Projects

This repository contains two projects that apply **machine learning and deep learning** to real-world problems:  

1. **Task 1: Email Spam Classification** – Detecting spam emails using classical machine learning models.  
2. **Task 2: MNIST Digit Recognition** – Recognizing handwritten digits using deep learning (CNN).  

Each project follows the machine learning workflow: **data preprocessing, visualization, model building, training, evaluation, and reporting**.  

---

## 🚀 Task 1: Email Spam Classification  

### 🎯 Objective  
Email spam detection is an essential application of machine learning in natural language processing (NLP). The goal is to **classify incoming emails as spam or not spam**, reducing unwanted content for users.  

### 🧠 Theoretical Background  
- **Spam filtering** is a **text classification problem**, where emails are treated as documents and represented as numerical feature vectors.  
- **Machine Learning Approach**:  
  - Emails are preprocessed to remove noise (stopwords, punctuation, special symbols).  
  - Features are extracted using text vectorization techniques (like Bag of Words or TF-IDF).  
  - Classification models (Random Forest, KNN) learn patterns that differentiate spam from non-spam emails.
  - <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/57a83ae3-ea69-4975-bb0f-0ba20ed82406" />
 

- **Algorithms Used**:  
  - **Random Forest**: An ensemble of decision trees that improves accuracy and reduces overfitting. Each tree votes, and the majority decides the final prediction.  
  - **K-Nearest Neighbors (KNN)**: A distance-based algorithm that classifies an email based on the majority class of its nearest neighbors.
  - <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/c24d850c-c962-4148-879f-ec21d26bc305" />


### 🛠 Implementation Steps  
1. **Data Preprocessing**  
   - Checked for missing/null values.  
   - Cleaned text data: removed stopwords, lowercased, tokenized.  
   - Transformed text into numerical features.  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized spam vs. non-spam email distribution.  
   - Analyzed word frequencies.  

3. **Model Training**  
   - Trained **Random Forest** and **KNN** classifiers.  
   - Compared performance on test data.  

4. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score.  
   - Confusion matrix for error analysis.  

### 📊 Results  
- Both models achieved good performance.  
- Random Forest showed **higher accuracy** and robustness compared to KNN.
- <img width="1040" height="784" alt="image" src="https://github.com/user-attachments/assets/1ec4ace1-07e4-42f3-8e03-033c1f3c1897" />

 

📂 Notebook: **`Task_1_Email_Spam_Classification.ipynb`** 
<img width="1889" height="886" alt="Screenshot 2025-08-13 121802" src="https://github.com/user-attachments/assets/9ddb48d0-febf-4362-90d9-a5fee26dd4bc" />


---

## 🚀 Task 2: MNIST Digit Recognition  

### 🎯 Objective  
The MNIST dataset is a benchmark in computer vision, consisting of **60,000 training images and 10,000 test images** of handwritten digits (0–9). The task is to build a model that can **recognize digits** with high accuracy.  
<img width="1475" height="677" alt="image" src="https://github.com/user-attachments/assets/8940ca53-ef0c-4f2d-bcb4-09185a3f200c" />
<img width="1058" height="747" alt="image" src="https://github.com/user-attachments/assets/9f8aa3ff-9424-438c-b768-5b8f5a1ee656" />


### 🧠 Theoretical Background  
- **Image Classification**: Images are represented as pixel intensity values. In MNIST, each image is **28x28 grayscale pixels**.  
- **Convolutional Neural Networks (CNNs)**: CNNs are the most effective models for image recognition because they:  
  - Use **convolution layers** to detect local patterns (edges, shapes).  
  - Apply **pooling layers** to reduce dimensionality and retain important features.  
  - Stack multiple convolution + pooling layers to learn hierarchical features.  
  - Use **fully connected layers** and a **softmax output layer** for classification.
  - <img width="924" height="570" alt="image" src="https://github.com/user-attachments/assets/d546722c-0c7e-43dc-ae37-63f147b9461b" />


- **Why CNN instead of traditional ML?**  
  Traditional ML requires manual feature extraction, but CNNs automatically learn feature representations from images, making them more powerful for vision tasks.  

### 🛠 Implementation Steps  
1. **Data Preprocessing**  
   - Normalized pixel values (0–255 → 0–1).  
   - One-hot encoded digit labels (0–9).  

2. **Visualization & Augmentation**  
   - Visualized sample digit images.  
   - Used **data augmentation** (rotations, shifts, zooms) to make the model more robust.  

3. **Model Training**  
   - Built a CNN with:  
     - Convolution + ReLU layers  
     - MaxPooling layers  
     - Dense (fully connected) layers  
     - Softmax output for 10 digits  
   - Optimized using Adam optimizer.
   - <img width="1396" height="508" alt="image" src="https://github.com/user-attachments/assets/5b6fc8ed-b155-4f2b-bc9d-90d8b41efe7e" />


4. **Evaluation**  
   - Accuracy and loss curves plotted for training & validation.  
   - Tested on unseen data.  

### 📊 Results  
- The CNN achieved **~99% accuracy** on the test set.  
- Data augmentation improved generalization.  
- MNIST recognition task demonstrates the effectiveness of CNNs in image classification.
- <img width="1782" height="729" alt="image" src="https://github.com/user-attachments/assets/ae84e8f4-3216-47a6-8c0f-3c65af9259fc" />
 <img width="1902" height="874" alt="Screenshot 2025-08-13 125300" src="https://github.com/user-attachments/assets/9292023f-417b-40b8-829d-a5ce2c6178c4" />

📂 Notebook: **`Task_2_Mnist_Digit_Recognition.ipynb`**  

---

## ⚙️ Technologies Used  
- **Python**  
- **Scikit-learn** → Random Forest, KNN (Spam classification)  
- **TensorFlow / Keras** → CNN (Digit recognition)  
- **NumPy & Pandas** → Data preprocessing  
- **Matplotlib & Seaborn** → Visualization  

---

## 📊 Performance Summary  

| Task | Model(s) Used | Best Accuracy |
|------|--------------|---------------|
| Email Spam Classification | Random Forest, KNN | ~Reported in notebook |
| MNIST Digit Recognition | CNN | **99%** |

---


📌 Conclusion

Task 1 shows the application of classical machine learning (Random Forest, KNN) in text classification problems.

Task 2 demonstrates the power of deep learning (CNNs) in computer vision, achieving 99% accuracy on MNIST.

Together, these projects cover two fundamental domains of ML: Natural Language Processing (NLP) and Computer Vision (CV).

This repository serves as a learning resource for applying ML/DL techniques across different data modalities.
