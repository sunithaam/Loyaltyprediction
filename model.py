import pandas as pd
import numpy as np
import pickle
import openpyxl
import joblib
from sklearn.model_selection import train_test_split
#Reading the data
df=pd.read_excel('F:/ICT_Inter/Electronic.xlsx')
pickle.dump(  df, open( "df.pkl", "wb" ))
df['Satisfaction Score']=df['Satisfaction Score'].astype('int64')
df[['Loyalty Score','Store Rating']]=df[['Loyalty Score','Store Rating']].round(1)     
#Handling missing values
df['Store Rating'].fillna(df['Store Rating'].median(),inplace=True)
from sklearn.impute import SimpleImputer
null_columns = df.columns[df.isnull().any()]
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer=imp_mode.fit(df[null_columns])
df_imp=imputer.transform(df[null_columns])
df_imp=pd.DataFrame(df_imp)
df_imp.columns=['Gender','Payment Method','Membership Status','Preferred Visit Time']
df[null_columns]=df_imp[null_columns]
#Discount and Total spent are correlated,so discount can be removed from model
# Create a list of the payment methods to check
#payment_methods = ['Cash', 'Debit Card', 'UPI']
# Use the isin() method to check if the 'Payment Method' is in the list
#df['PM'] = df['Payment Method'].isin(payment_methods).astype(int)
df_new=df.drop(['Discount (%)'],axis=1)
df_new.drop(['Age','Gender','Region'],axis=1,inplace=True)
df_new['Membership Status']=df_new['Membership Status'].astype('int64')
#Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
# Fit and transform the categorical data 
encoded_data = encoder.fit_transform(df_new[['Product Category','Payment Method','Preferred Visit Time']])
# Convert the encoded data to a DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Product Category','Payment Method','Preferred Visit Time']))
# Concatenate with the original DataFrame 
df_encoded = pd.concat([df_new, encoded_df], axis=1).drop(columns=['Product Category','Payment Method','Preferred Visit Time']) 
# Separate features and target variable 
X = df_encoded.drop('Loyalty Score',axis=1)
y = df_encoded['Loyalty Score']
#Splitting the data based on selected features
X=X[['Items Purchased','Total Spent','Satisfaction Score','Warranty Extension','Revenue','Payment Method_Cash']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Model building using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200, random_state=42,bootstrap=True,max_depth= 10,max_features= 'sqrt',min_samples_leaf= 2,min_samples_split= 10)

# Train the model 
rf.fit(X_train, y_train)
with open('model.pkl','wb') as model_file:
    pickle.dump(rf,model_file)