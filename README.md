# Boston House Price Prediction

This project is a Flask-based web application that predicts house prices in the Boston area using a Linear Regression model. The application uses a pre-trained model to provide estimates based on user-provided features. The model is trained on the historical Boston Housing dataset, and it's important to acknowledge the ethical concerns associated with certain features within that dataset.

## Table of Contents

1.  [Ethical Considerations](#ethical-considerations)
2.  [Project Overview](#project-overview)
3.  [Prerequisites](#prerequisites)
4.  [Installation](#installation)
5.  [Running the Application](#running-the-application)
6.  [Model Training](#model-training)
7.  [Feature Imputation](#feature-imputation)
8.  [Code Structure](#code-structure)
9.  [Data Source](#data-source)
10. [License](#license)
11. [Contact](#contact)

## Ethical Considerations

This project uses the Boston Housing dataset, which contains features that are now recognized as ethically problematic. Specifically:

*   **"Proportion of Blacks by Town" (B):** Engineered with the assumption that racial self-segregation had a positive impact on house prices, reflecting racist and discriminatory biases.
*   **"Lower Status of the Population" (LSTAT):** Can perpetuate and reinforce harmful biases related to socioeconomic status, as it's often correlated with protected characteristics.

Steps have been taken to mitigate these ethical concerns:

*   Removed problematic features from the user interface.
*   Imputing missing values using KNN.
*   Acknowledging the limitations of the dataset.

Users are strongly encouraged to consider the ethical implications of using this dataset and to explore alternative, more ethically sound datasets whenever possible.

## Project Overview

The application allows users to input house features, such as crime rate, residential zoning, number of rooms, and air quality, to obtain a predicted house price. The prediction is made using a Linear Regression model trained on the Boston Housing dataset.

## Prerequisites

Before running the application, ensure you have the following installed:

*   Python (3.7 or higher)
*   pip
*   Virtualenv (optional): `pip install virtualenv`
*   Git: To clone this project `git clone`

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/RamharshDandekar/DevTownSubmission.git
    cd DevTownSubmission
    ```

2.  Create a virtual environment (recommended):

    ```bash
    virtualenv venv
    ```

3.  Activate the virtual environment:

    *   On Windows:

        ```bash
        venv\Scripts\activate
        ```

    *   On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

4.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```


## Running the Application

1.  Train the Model:

    *   Open the `Boston_House_Price_Prediction.ipynb` notebook.
    *   Run all cells to train the model and save the `.pkl` file.

2.  Run the Flask App:

    *   Navigate to the project directory where `app.py` is located.
    *   Run the Flask app:

        ```bash
        python app.py
        ```

3.  Access the Application:

    *   Open your web browser and go to `http://127.0.0.1:5000/`.

4.  Enter Input Values and click "Predict Price".

## Model Training

The Linear Regression model is trained using the Boston Housing dataset. The `Boston_House_Price_Prediction.ipynb` notebook contains the code for loading, preprocessing, splitting, training, evaluating, and saving the model. Data is sourced from "lib.stat.cmu.edu/datasets/boston".

## Feature Imputation

k-Nearest Neighbors (k-NN) is used to impute values.  This is done using:
*vector
*scale
*most similar to make project the best

## Code Structure

```
BostonPricePrediction/
├── Boston_House_Price_Prediction.ipynb
├── app.py
├── boston_model.pkl
├── requirements.txt
├── templates/
│ └── index.html
└── static/
└── style.css
└── .gitignore
└── README.md
```
## Data Source

The Boston Housing dataset is sourced from:
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

http://lib.stat.cmu.edu/datasets/boston

This dataset has ethical concerns.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please contact [Ramharsh Sanjay Dandekar](https://www.linkedin.com/in/ramharsh-sanjay-dandekar).
