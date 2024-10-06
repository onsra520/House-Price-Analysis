import os, glob, joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def Model_Saving(linear_model, scaler):
    Main_Path = os.getcwd()
    if os.path.exists('Model Results'):
        files = glob.glob(os.path.join('Model Results', '*'))
        for f in files:
            os.remove(f)
    else:
        os.makedirs("Model Results", exist_ok=True) 
    os.chdir(os.path.join(Main_Path, 'Model Results'))        
    joblib.dump(linear_model,'Price_Prediction Model.pkl',)
    joblib.dump(scaler, 'Scaler.pkl')
    os.chdir(Main_Path)

def Model_Predict(House_List):
    Location_List = [
        "Quận 1", "Quận 2", "Quận 3", "Quận 4", "Quận 5", "Quận 6", "Quận 7", "Quận 8", "Quận 9",
        "Quận 10", "Quận 11", "Quận 12", "Bình Thạnh", "Gò Vấp", "Phú Nhuận", "Tân Bình",
        "Tân Phú", "Bình Tân", "Thủ Đức", "Nhà Bè", "Hóc Môn", "Bình Chánh", "Củ Chi", "Cần Giờ" 
    ]

    # Ánh xạ các khu vực thành số
    Location_Mapping = {location: index for index, location in enumerate(Location_List)}
    Inverse_Location_Mapping = {index: location for location, index in Location_Mapping.items()}

    House_List = House_List.dropna() 
    
    # Chuyển hóa dữ liệu thành dạng số cho model
    House_List["Location"] = House_List["Location"].map(Location_Mapping)

    X = House_List[["Area", "Bedrooms", "Bathrooms", "Location"]]
    Y = House_List["Price"]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Huấn luyện model 
    linear_model = LinearRegression()
    linear_model.fit(X_train, Y_train)

    # Dự đoán trên tập huấn luyện và tập kiểm tra
    Y_pred_train = linear_model.predict(X_train)
    Y_pred_test = linear_model.predict(X_test)

    Test_Location = House_List.loc[Y_test.index, 'Location']
    Test_Area = House_List.loc[Y_test.index, 'Area']
    Test_Bedrooms = House_List.loc[Y_test.index, "Bedrooms"]
    Test_Bathrooms = House_List.loc[Y_test.index, "Bathrooms"]

    Results = pd.DataFrame({
        'Location': Test_Location,
        'Area': Test_Area,
        'Bedrooms': Test_Bathrooms,
        'Bathrooms': Test_Bedrooms,
        'Actual': Y_test,
        'Predicted': Y_pred_test
    })

    Results['Location'] = Results['Location'].map(Inverse_Location_Mapping)

    Results['Trend Prediction'] = np.where(Results['Predicted'].diff() > 0, 'Increase', 'Decrease')
    Results['Trend Actual'] = np.where(Results['Actual'].diff() > 0, 'Increase', 'Decrease')
    
    Model_Saving(linear_model, scaler)
    return Results.reset_index(drop = True)