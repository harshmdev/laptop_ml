import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import joblib , hyperopt


df1=pd.read_csv("data/processed/after_feature_selection.csv")
df=df1.copy()


df.drop(columns=['ppi_type',"quality_type"],inplace=True)



X=df.drop("price",axis=1)
y=df["price"]
y_transformed=np.log1p(y)


ohe_columns=["brand","graphics_brand","processor_brand","touch_screen","backlit","business","gaming"]
oe_columns=["thickness","typec","screen_size","processor_gen","processor_model","popularity","graphics_model"]
std=["ppi","battery_capacity","ssd","threads","ram","cores","graphics_capacity","battery_cell"]




# Creating a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), std),
        ('cat', OrdinalEncoder(), oe_columns),
        ('cat1',OneHotEncoder(drop="first",sparse_output=False),ohe_columns)
    ], 
    remainder='passthrough'
)


transformed_data=preprocessor.fit_transform(X)
transformed_df = pd.DataFrame(transformed_data, columns=preprocessor.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(transformed_df,y_transformed,test_size=0.2,random_state=42)



from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def objective(params):
    # Convert parameters to the correct types
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth']) if params['max_depth'] is not None else None
    params['min_samples_split'] = int(params['min_samples_split'])
    params['min_samples_leaf'] = int(params['min_samples_leaf'])

    # Create the Extra Trees model
    model = ExtraTreesRegressor(**params, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    preds = model.predict(X_test)
    preds=np.expm1(preds)
    
    # Calculate the mean squared error
    mse = mean_absolute_error(np.expm1(y_test), preds)
    
    # Return the loss and the status
    return {'loss': mse, 'status': STATUS_OK}


# In[68]:


space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2']),
    'max_depth': hp.quniform('max_depth', 5, 50, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 20, 1),
    'bootstrap': hp.choice('bootstrap', [False, True])
}


# In[69]:


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best hyperparameters:", best)


# In[73]:


# Convert the best hyperparameters to the correct types
best['n_estimators'] = int(best['n_estimators'])
best['max_depth'] = int(best['max_depth']) if best['max_depth'] is not None else None
best['min_samples_split'] = int(best['min_samples_split'])
best['min_samples_leaf'] = int(best['min_samples_leaf'])

# Handle None type for max_features and bootstrap correctly
if best['max_features'] is not None:
    best['max_features'] = [None, 'sqrt', 'log2'][int(best['max_features'])]

best['bootstrap'] = [False, True][int(best['bootstrap'])]

# Create the Extra Trees model with the best hyperparameters
best_model = ExtraTreesRegressor(**best, random_state=42)

# Train the model
best_model.fit(X_train, y_train)

# Make predictions
final_preds = best_model.predict(X_test)
final_preds = np.expm1(final_preds)

# Evaluate the model
final_mae = mean_absolute_error(np.expm1(y_test), final_preds)
final_r2 = r2_score(np.expm1(y_test), final_preds)

print("Final MAE:", final_mae)
print("Final R²:", final_r2)



# Export the model using joblib
joblib_file = "model/model.pkl"
joblib.dump(best_model, joblib_file)

print(f"Model saved to {joblib_file}")


# In[95]:


# Load the model from the file
loaded_model = joblib.load(joblib_file)

# Use the loaded model to make predictions
loaded_preds = loaded_model.predict(X_test)
loaded_preds=np.expm1(loaded_preds)

# Evaluate the loaded model
loaded_mse = mean_absolute_error(np.expm1(y_test), loaded_preds)
loaded_r2 = r2_score(np.expm1(y_test), loaded_preds)

print("Loaded Model MSE:", loaded_mse)
print("Loaded Model R²:", loaded_r2)
