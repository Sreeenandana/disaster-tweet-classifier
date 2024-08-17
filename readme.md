
# DTWEET: A Disaster Tweet Classifier 

## Overview

The Disaster Tweet Recognizer project utilizes Machine Learning (ML) and Natural Language Processing (NLP) techniques to classify tweets as either genuinely related to disasters or not. This project aims to improve the efficiency of disaster management by automatically filtering relevant tweets from a large volume of social media data.

Additionally, the project includes a user-friendly web application built with Streamlit, allowing users to input tweets and get immediate classification results.

## Features

- **Tweet Classification**: Uses ML models and NLP techniques to categorize tweets into "Disaster" or "Not Disaster."
- **Streamlit Web App**: An interactive web interface where users can input tweets and view classification results.

## Technologies Used

- **Machine Learning**: For classification model training and prediction.
- **Natural Language Processing (NLP)**: For text preprocessing and feature extraction.
- **Streamlit**: For building the interactive web application.
- **Python Libraries**: 
  - `scikit-learn` for ML algorithms.
  - `nltk` and `spaCy` for NLP.
  - `pandas` and `numpy` for data manipulation.

## Getting Started

### Prerequisites

- Python 3.x
- `pip` (Python package installer)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/dtweet.git
   cd dtweet
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Train the Model**:
   - Run the training script to train the ML model:
     ```bash
     python train_model.py
     ```

2. **Start the Streamlit Web App**:
   ```bash
   streamlit run app.py
   ```

   This will launch the Streamlit web application in your browser.

### Usage

1. Open the web application in your browser.
2. Enter a tweet into the input field.
3. Click "Classify" to see whether the tweet is classified as "Disaster" or "Not Disaster."


## Acknowledgements

- [Streamlit](https://streamlit.io/) for creating a powerful tool for building web apps.
- [scikit-learn](https://scikit-learn.org/) for providing ML algorithms and tools.

---

Feel free to adjust the content based on the actual structure and details of your project.
