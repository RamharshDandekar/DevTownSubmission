from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the model, scaler, poly, KNN, feature_list
try:
    model, scaler, poly, knn, features_for_nn, poly_knn, scaler_knn = pickle.load(open('boston_model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'boston_model.pkl' not found.")
    exit()

# --- Mapping Functions ---
def map_air_quality_to_nox(quality):
    if quality == "Very Good":
        return 0.3
    elif quality == "Good":
        return 0.4
    elif quality == "Fair":
        return 0.5
    elif quality == "Poor":
        return 0.6
    elif quality == "Very Poor":
        return 0.7
    else:
        return 0.5  # Default

def map_accessibility_to_dis(accessibility):
    return accessibility * 0.5 + 1

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def impute_b_lstat(input_data, knn, features_for_nn, df_original,poly_knn,scaler_knn):
    """Imputes B and LSTAT values using KNN."""
    # Create a DataFrame from the input features, matching the structure used to train KNN

    input_df = pd.DataFrame([input_data]) # New
    input_df.columns = features_for_nn  # Manually assign column names

    # scaling and polynomia must be done.
    #Scale Data first because you fit on training data
    input_poly = poly_knn.transform(input_df)
    input_scaled = scaler_knn.transform(input_poly)

    # Find the nearest neighbors in the original dataset
    distances, indices = knn.kneighbors(input_scaled)

    # Average B and LSTAT values from the nearest neighbors
    nearest_neighbors = df_original.iloc[indices[0]]
    imputed_b = nearest_neighbors['B'].mean()
    imputed_lstat = nearest_neighbors['LSTAT'].mean()

    return imputed_b, imputed_lstat

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user-friendly feature values from the form
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])

            air_quality = request.form['air_quality']
            NOX = map_air_quality_to_nox(air_quality)

            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])

            accessibility = float(request.form['accessibility'])
            DIS = map_accessibility_to_dis(accessibility)

            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])

            # Prepare input data as a dictionary for KNN
            input_data = {
                "CRIM": CRIM, "ZN": ZN, "INDUS": INDUS, "CHAS": CHAS, "NOX": NOX,
                "RM": RM, "AGE": AGE, "DIS": DIS, "RAD": RAD, "TAX": TAX, "PTRATIO": PTRATIO
            }
            #Load data here
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)  # Fixed SyntaxWarning
            data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]

            # Dataframe for KNN only
            df_original = pd.DataFrame(data, columns=[
                "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
            ])
            df_original['target'] = target


            # Impute B and LSTAT values
            B, LSTAT = impute_b_lstat(input_data, knn, features_for_nn, df_original,poly_knn,scaler_knn) # KNN Impute

            # Create a DataFrame for the original features
            input_data = pd.DataFrame([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]],
                                      columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

            # Apply polynomial feature transformation
            input_poly = poly.transform(input_data)
            input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(input_features=input_data.columns))

            # Scale the transformed input data
            input_scaled = scaler.transform(input_poly_df)

            # Make the prediction
            prediction = model.predict(input_scaled)
            output = round(prediction[0], 2)

            return render_template('index.html', prediction_text='Predicted House Price: $ {}'.format(output))

        except ValueError:
            return render_template('index.html', prediction_text='Please enter valid numerical values.')
        except Exception as e:
            return render_template('index.html', prediction_text=f'An error occurred: {str(e)}')
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)