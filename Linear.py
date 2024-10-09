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
import seaborn as sns

##Data inladen

df1 = pd.read_csv("airports-extended-clean.csv", sep=';')
df2 = pd.read_csv("case3_data/schedule_airport.csv")
#Kolomnaam veranderen 
df2 = df2.rename(columns={'STD' : 'date', 'STA_STD_ltc' : 'gepl_aank', 'ATA_ATD_ltc' : 'werk_aank'})
#Dataset merge, op luchthaven/code
df3 = df2.merge(df1[['ICAO', 'Name']], left_on='Org/Des', right_on='ICAO', how='left')


# Zet de datum- en tijdstempels om naar datetime-formaat
df3['gepl_aank'] = pd.to_datetime(df3['date'] + ' ' + df3['gepl_aank'], format='%d/%m/%Y %H:%M:%S')
df3['werk_aank'] = pd.to_datetime(df3['date'] + ' ' + df3['werk_aank'], format='%d/%m/%Y %H:%M:%S')

# Bereken de vertraging (in minuten)
df3['vertraging'] = (df3['werk_aank'] - df3['gepl_aank']).dt.total_seconds() /60


# Definieer features en target
X = pd.get_dummies(df3['Name'], drop_first=True)  # One-hot encoding van bestemmingen
y = df3['vertraging']

# Splits de data in training en test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train het model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title("Vertraging Voorspelling per Bestemming")

# Selecteer de bestemming
bestemmingen = df3['Name'].unique()
keuze = st.selectbox("Kies een bestemming:", bestemmingen)

# Voorspel vertraging voor de gekozen bestemming
if keuze:
    # Maak een input vector voor de voorspelling
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0  # Start met een rij van nullen
    input_data[keuze] = 1  # Zet de gekozen bestemming op 1

    # Voorspel de vertraging
    voorspelling = model.predict(input_data)[0]
    st.write(f"Verwachte vertraging voor {keuze}: {voorspelling:.2f} seconden")

df3['date'] = pd.to_datetime(df3['date'], format='%d/%m/%Y')

# Voeg maand kolom toe
df3['month'] = df3['date'].dt.month

# Dropdown menu om jaar te kiezen
year = st.selectbox('Kies een jaar:', [2019, 2020])

# Filter de DataFrame op het gekozen jaar
df_filtered = df3[df3['date'].dt.year == year]

# Bereken de gemiddelde vertraging per maand
avg_delay_per_month = df_filtered.groupby('month')['vertraging'].mean()

# Maak de grafiek
plt.figure(figsize=(10, 6))
avg_delay_per_month.plot(kind='bar', color='skyblue')
plt.title(f'Gemiddelde Vertraging per Maand in {year}')
plt.xlabel('Maand')
plt.ylabel('Gemiddelde Vertraging (seconden)')
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'], rotation=45)
plt.grid(axis='y')
st.pyplot(plt)

# Voeg kolommen toe voor maand en jaar
df3['date'] = pd.to_datetime(df3['date'], format='%d/%m/%Y')
df3['month'] = df3['date'].dt.month
df3['year'] = df3['date'].dt.year
df3['season'] = df3['month'].apply(lambda x: 'Lente' if x in [3, 4, 5] else
                                       ('Zomer' if x in [6, 7, 8] else
                                        ('Herfst' if x in [9, 10, 11] else 'Winter')))

# Streamlit interface
st.title('Vluchtvertraging Voorspelling en Analyse')

# Vliegperiode selectie
selected_season = st.selectbox('Selecteer het seizoen:', ['Lente', 'Zomer', 'Herfst', 'Winter'])
selected_month = st.selectbox('Selecteer de maand:', range(1, 13))
selected_year = st.selectbox('Selecteer het jaar:', [2019, 2020])

# Filter de data op basis van de geselecteerde periode
df_filtered = df3[(df3['season'] == selected_season) | (df3['month'] == selected_month)]
df_filtered = df_filtered[df_filtered['year'] == selected_year]

# Visualisatie van gemiddelde vertraging per maand
avg_delay = df_filtered.groupby('month')['vertraging'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_delay.index, y=avg_delay.values, palette='viridis')
plt.title('Gemiddelde Vertraging per Maand')
plt.xlabel('Maand')
plt.ylabel('Gemiddelde Vertraging (seconden)')
plt.xticks(rotation=45)
st.pyplot(plt)

# Boxplot van de vertraging per jaar
plt.figure(figsize=(10, 6))
sns.boxplot(x='year', y='vertraging', data=df3)
plt.title('Boxplot van Vertraging per Jaar')
plt.xlabel('Jaar')
plt.ylabel('Vertraging (seconden)')
plt.xticks(rotation=45)
st.pyplot(plt)

