# 🖼️ AI Image Caption Generator

This project is a web-based application built with Streamlit that uses a deep learning model to automatically generate descriptive captions for images. It leverages a combination of a Convolutional Neural Network (CNN) and a Long Short-Term Memory (LSTM) network to analyze visual content and translate it into human-readable text.


---

## ✨ Features

-   **Intuitive Web Interface**: A clean and user-friendly interface powered by Streamlit.
-   **AI-Powered Captioning**: Utilizes a sophisticated deep learning model to generate accurate and contextually relevant captions.
-   **Real-time Processing**: Upload an image and get a caption in seconds.
-   **Informative UI**: Includes details about the underlying technology and model architecture.
-   **Downloadable Captions**: Easily download the generated caption as a text file.

---

## 🚀 How It Works

The model's architecture consists of two main components working in tandem:

1.  **Encoder (Image Understanding)**:
    -   A **Convolutional Neural Network (CNN)**, specifically the pre-trained **DenseNet-201** model, is used as the image feature extractor.
    -   The CNN scans the input image and identifies key objects, patterns, and scenes, converting this visual information into a dense vector representation (an "embedding").

2.  **Decoder (Caption Generation)**:
    -   A **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN), takes the image's vector representation from the encoder.
    -   It then generates the caption word by word, learning the sequence and structure of language to form a coherent sentence that describes the image.

The model was trained on the **Flickr8k dataset**, which contains thousands of images, each paired with multiple descriptive captions.

---

## 🛠️ Setup and Installation

To run this project locally, follow these steps:

**1. Clone the Repository**

```bash
git clone https://github.com/Aafimalek/image-captioning.git
cd image-captioning
```

**2. Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

Install the required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**4. Download Model Files**

The deep learning model files are too large for this repository and are ignored by `.gitignore`. You will need to download them separately and place them in a `models/` directory at the root of the project.

Create the directory:
```bash
mkdir models
```
*Make sure you have `model.keras`, `tokenizer.pkl`, and `feature_extractor.keras` inside the `models/` folder.*

---

## 🏃‍♀️ Usage

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser. From there, you can upload an image to start generating captions.

---

## ⚙️ Technology Stack

-   **Backend**: Python
-   **Deep Learning**: TensorFlow, Keras
-   **Web Framework**: Streamlit
-   **Data Handling**: NumPy, Pillow
-   **Dataset**: Flickr8k

---

## 🙏 Acknowledgments

-   This project was inspired by the many great resources available on image captioning and deep learning.
-   Special thanks to the creators of the Flickr8k dataset for providing valuable data for training. 