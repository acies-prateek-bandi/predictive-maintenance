import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler


def preprocess_data(iot_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    iot_data = pd.concat([iot_data,pd.get_dummies(iot_data['Type'])],axis=1)
    iot_data.drop(["UDI","Product ID","Type"], axis=1, errors="ignore",inplace=True)
    
   
    return iot_data, {"columns": iot_data.describe().T.to_dict(), "data_type": "iot_data"}




