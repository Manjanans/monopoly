import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import QuantileTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,classification_report,accuracy_score


def multicolumn_IQR_dropna(dataframe:pd.DataFrame) -> pd.DataFrame:
    """
    This function does the following:

    1. Copies the ID, age, sex and Income columns
    2. Calculates the IQR for each column
    3. Filter out the outliers for the current column only
    4. Merge filtered column back into a DataFrame, using an 'ID' column

    Args:
        dataframe (pd.DataFrame): DataFrame to perform IQR

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    dato = pd.DataFrame()
    for columna in dataframe.columns:
        Q1 = dataframe[columna].quantile(0.25)
        Q3 = dataframe[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers for the current column only
        col_limpia = dataframe[~((dataframe[columna] < lower_bound) | (dataframe[columna] > upper_bound))][[columna]]
        # Merge filtered column back into the 'dato' DataFrame on the 'ID' column
        dato[columna] = col_limpia
    dato = dato.dropna()
    return dato

def multicolumn_IQR(dataframe:pd.DataFrame) -> pd.DataFrame:
    """
    This function does the following:

    1. Copies the ID, age, sex and Income columns
    2. Calculates the IQR for each column
    3. Filter out the outliers for the current column only
    4. Merge filtered column back into a DataFrame, using an 'ID' column

    Args:
        dataframe (pd.DataFrame): DataFrame to perform IQR

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    dato = pd.DataFrame()
    for columna in dataframe.columns:
        Q1 = dataframe[columna].quantile(0.25)
        Q3 = dataframe[columna].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers for the current column only
        col_limpia = dataframe[~((dataframe[columna] < lower_bound) | (dataframe[columna] > upper_bound))][[columna]]
        # Merge filtered column back into the 'dato' DataFrame on the 'ID' column
        dato[columna] = col_limpia
    return dato

def clean_data(client_data:pd.DataFrame, save_path_international, save_path_acc, save_path_predict_international, save_path_predict_num_acc) -> pd.DataFrame:
    selected_features = ['Num_CC', 'National', 'International', 'Num_Acc']
    df = pd.DataFrame(client_data[selected_features])
    df_iqr = multicolumn_IQR(df)
    knn_imputer = KNNImputer(n_neighbors=10, weights='uniform')
    imputed = pd.DataFrame(knn_imputer.fit_transform(df_iqr), columns=df_iqr.columns)
    imputed['Num_Acc'] = imputed['Num_Acc'].replace({1: 0, 2: 1, 3: 1})
    quantile = QuantileTransformer(output_distribution='normal')
    quantile.fit(imputed)
    df_quantile = pd.DataFrame(quantile.transform(imputed), columns=imputed.columns)
    pik = QuantileTransformer(output_distribution='normal')
    pok = QuantileTransformer(output_distribution='normal')
    tik = QuantileTransformer(output_distribution='normal')
    tok = QuantileTransformer(output_distribution='normal')
    international=pik.fit(imputed[['International']])
    num_acc = pok.fit(imputed[['Num_Acc']])
    predict_international = tik.fit(imputed[['Num_Acc','National']])
    predict_num_acc = tok.fit(imputed[['Num_CC', 'National', 'International']])
    final = multicolumn_IQR_dropna(df_quantile)
    
    with open(save_path_international, 'wb') as f:
        pickle.dump(international, f)
    with open(save_path_acc, 'wb') as f:
        pickle.dump(num_acc, f)
    with open(save_path_predict_international, 'wb') as f:
        pickle.dump(predict_international, f)
    with open(save_path_predict_num_acc, 'wb') as f:
        pickle.dump(predict_num_acc, f)

    return final

def international_model(clean_data:pd.DataFrame, save_path_international_model) -> pd.DataFrame:
    # --- Carga de datos y selección de características ---
    # Supongamos que 'dat' es el DataFrame con las variables relevantes.
    dat = pd.DataFrame()
    dat['Num_Acc'] = clean_data['Num_Acc']  
    dat['National'] = clean_data['National']
    dat['International'] = clean_data['International']

    # --- División de datos en entrenamiento, validación y prueba ---
    train, valid, test = np.split(dat.sample(frac=1, random_state=42), [int(0.6*len(dat)), int(0.8*len(dat))])

    X_train = train.drop(columns=['International'])
    y_train = train['International']

    X_valid = valid.drop(columns=['International'])
    y_valid = valid['International']

    X_test = test.drop(columns=['International'])
    y_test = test['International']

    # --- Creación del modelo ---
    model = XGBRegressor(
        booster='dart',               # Tipo de booster
        learning_rate=0.2,            # Tasa de aprendizaje
        max_depth=3,                  # Profundidad máxima del árbol
        n_estimators=100,             # Número de árboles
        objective='reg:squarederror', # Objetivo de regresión
        tree_method='hist',           # Uso de GPU
        random_state=42               # Reproducibilidad
    )

    # --- Entrenamiento del modelo ---
    model.fit(X_train, y_train)
    y_pred_valid = model.predict(X_valid)
    y_pred_test = model.predict(X_test)

    # Métricas para validación
    mse_valid = mean_squared_error(y_valid, y_pred_valid)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    r2_valid = r2_score(y_valid, y_pred_valid)
    rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=False)

    # Métricas para prueba
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    # --- Impresión de resultados ---
    print("Resultados de Validación:")
    print(f"Mean Squared Error: {mse_valid}")
    print(f"Mean Absolute Error: {mae_valid}")
    print(f"R^2: {r2_valid}")
    print(f"Root Mean Squared Error: {rmse_valid}\n")

    print("Resultados de Prueba:")
    print(f"Mean Squared Error: {mse_test}")
    print(f"Mean Absolute Error: {mae_test}")
    print(f"R^2: {r2_test}")
    print(f"Root Mean Squared Error: {rmse_test}\n")

    # Puntaje general del modelo en el conjunto de validación
    print(f"Puntaje del modelo en validación: {model.score(X_valid, y_valid)}")

    with open(save_path_international_model, 'wb') as f:
        pickle.dump(model, f)

def num_acc_model(clean_data:pd.DataFrame, save_path_num_acc_model):
    # Split the data into features (X) and target (y)
    X = clean_data[['Num_CC', 'National', 'International']]
    y = clean_data[['Num_Acc']].astype('int')
    
    
    # Split data into train (70%) and temp (30%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Further split temp into validation (50% of temp) and test (50% of temp)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Balance the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Initialize and train the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on validation data
    y_valid_pred = rf.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    print("\nValidation Classification Report:")
    print(classification_report(y_valid, y_valid_pred))
    
    # Evaluate on test data
    y_test_pred = rf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Results summary
    results = {
        'Validation Accuracy': valid_accuracy,
        'Test Accuracy': test_accuracy
    }

    with open(save_path_num_acc_model, 'wb') as f:
        pickle.dump(rf, f)
