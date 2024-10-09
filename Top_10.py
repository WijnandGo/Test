import folium.map
import plotly.express as px
import pandas as pd
import streamlit as st
import folium
import matplotlib.pyplot as plt
import branca
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

##Data inladen

df1 = pd.read_csv(r"C:\Users\goedh\OneDrive\Minor\Case 3\airports-extended-clean.csv",sep=';')
df2 = pd.read_csv(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\schedule_airport.csv")

#Kolomnaam veranderen 
df2 = df2.rename(columns={'STD' : 'date', 'STA_STD_ltc' : 'gepl_aank', 'ATA_ATD_ltc' : 'werk_aank'})
#Dataset merge, op luchthaven/code
df3 = df2.merge(df1[['ICAO', 'Name']], left_on='Org/Des', right_on='ICAO', how='left')

# Converteer de datum naar datetime-formaat
df3['date'] = pd.to_datetime(df3['date'], format='%d/%m/%Y')

# Voeg een nieuwe kolom toe voor de maand
df3['month'] = df3['date'].dt.month

# Voeg een nieuwe kolom toe voor het seizoen
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Lente'
    elif month in [6, 7, 8]:
        return 'Zomer'
    else:
        return 'Herfst'

df3['season'] = df3['month'].apply(get_season)

# Bereken de vertraging in seconden
df3['vertraging'] = (pd.to_datetime(df3['werk_aank'], format='%H:%M:%S') - pd.to_datetime(df3['gepl_aank'], format='%H:%M:%S')).dt.total_seconds()

# Voorbereiden van de gegevens voor het model
X = pd.get_dummies(df3[['Name', 'month', 'season']], drop_first=True)  # One-hot encoding
y = df3['vertraging'] / 60  # Verandering in seconden naar minuten

# Train-test splitsing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Maak en train het model
model = LinearRegression()
model.fit(X_train, y_train)

# Maak een Streamlit-app
st.title("Vertraging Voorspellingsmodel")

# Dropdown voor het kiezen van de bestemming
selected_destination = st.selectbox("Kies je bestemming:", df3['Name'].unique())
selected_month = st.selectbox("Kies de maand:", range(1, 13))  # Maanden van 1 tot 12
selected_season = get_season(selected_month)  # Bepaal het seizoen

# Maak de invoer voor het model
input_data = pd.DataFrame({
    'Name': [selected_destination],
    'month': [selected_month],
    'season': [selected_season]
})

# One-hot encoding van de invoerdata
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

# Zorg ervoor dat de inputdata dezelfde kolommen heeft als de training data
for col in X.columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Voorspel de vertraging
expected_delay = model.predict(input_data_encoded[X.columns])

# Toon de verwachte vertraging in minuten
st.write(f"Verwachte vertraging voor vlucht naar {selected_destination} in {selected_season}: {expected_delay[0]:.2f} minuten.")

