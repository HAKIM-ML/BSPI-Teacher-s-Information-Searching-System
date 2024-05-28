# BSPI-Teacher's-Information-Searching-System

## Project Overview

The BSPI-Teacher's-Information-Searching-System is designed to identify teachers from images and display their information. This system is built using a dataset of 21,000 images of 21 teachers from BSPI University. The data has been preprocessed, and a transfer learning technique has been employed to classify the images. A Streamlit application serves as the user interface, allowing users to upload an image and receive detailed information about the identified teacher.

## Dataset

The dataset was collected from the BSPI University website and consists of 21,000 images of 21 teachers. The dataset has been uploaded to Kaggle and is available for further use and experimentation.

## Preprocessing

The data preprocessing steps include:
1. Cleaning the images.
2. Normalizing the image data.
3. Augmenting the images to enhance the dataset.
4. Splitting the dataset into training and testing sets.

## Model

A transfer learning technique was used to build the model. The steps involved are:
1. Using a pre-trained Convolutional Neural Network (CNN) model.
2. Fine-tuning the model with our dataset to classify the 21 teachers.
3. Evaluating the model's performance and optimizing it for better accuracy.

## Streamlit Application

The Streamlit app is the front-end of this project. Users can interact with the app by uploading an image, which the model then classifies. The app displays the identified teacher's information, including their name, department, and other relevant details.

### Features

- **Upload Image**: Users can upload an image of a teacher.
- **Classify Teacher**: The model processes the image and identifies the teacher.
- **Display Information**: The app displays the teacher's information based on the classification.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/HAKIM-ML/BSPI-Teacher-Information-Searching-System.git
    ```

2. Navigate to the project directory:
    ```
    cd BSPI-Teacher-Information-Searching-System
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```
    streamlit run app.py
    ```

## Usage

1. Open the Streamlit app in your browser.
2. Upload an image of a teacher from BSPI University.
3. Wait for the model to classify the image.
4. View the displayed information about the identified teacher.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.


## Acknowledgements

- Thanks to BSPI University for providing the dataset.
- Thanks to the developers of Streamlit and the creators of the pre-trained models used in this project.

