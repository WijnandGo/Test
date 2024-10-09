import folium.map
import plotly.express as px
import pandas as pd
import streamlit as st
import folium
import matplotlib.pyplot as plt
import branca
from streamlit_folium import st_folium

##Data inladen

df1 = pd.read_csv(r"C:\Users\goedh\OneDrive\Minor\Case 3\airports-extended-clean.csv",sep=';')
df2 = pd.read_csv(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\schedule_airport.csv")
df2 = df2.rename(columns={'STD' : 'date', 'STA_STD_ltc' : 'gepl_aank', 'ATA_ATD_ltc' : 'werk_aank'})
#Dataset merge, op luchthaven/code
df3 = df2.merge(df1[['ICAO', 'Name']], left_on='Org/Des', right_on='ICAO', how='left')




#Vlucht data
vlucht30_1 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 1.xlsx")
vlucht30_2 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 2.xlsx")
vlucht30_3 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 3.xlsx")
vlucht30_4 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 4.xlsx")
vlucht30_5 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 5.xlsx")
vlucht30_6 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 6.xlsx")
vlucht30_7 = pd.read_excel(r"C:\Users\goedh\OneDrive\Minor\Case 3\case3_data\case3\30Flight 7.xlsx")



'''
Deze code is voor de kaart met vluchroute incl altitudes en start/landingsbanen
'''

#Dicationary vluchten:
vluchten = {
    "Vlucht 1": vlucht30_1,
    "Vlucht 2": vlucht30_2,
    "Vlucht 3": vlucht30_3,
    "Vlucht 4": vlucht30_4,
    "Vlucht 5": vlucht30_5,
    "Vlucht 6": vlucht30_6,
    "Vlucht 7": vlucht30_7,
}

# Definieer de coördinaten voor de banen
banen = {
    "Polderbaan": (52.348250, 4.711250),  
    "Kaagbaan": (52.289852, 4.742564),
    "Aalsmeerbaan": (52.296685, 4.778077),
    "Oostbaan": (52.308029, 4.794181),
    "Zwanenburgbaan": (52.316599, 4.738689),
    "Buitenveldertbaan": (52.317653, 4.769317),
}

# Dropdown menu voor vlucht selectie
selected_vlucht = st.selectbox("Kies een vlucht:", list(vluchten.keys()))
df = vluchten[selected_vlucht]

# Verwijder rijen met NaN-waarden in de benodigde kolommen
df = df.dropna(subset=['[3d Latitude]', '[3d Longitude]', '[3d Altitude M]', 'Time (secs)'])

if df.empty:
    st.error("Geen geldige gegevens beschikbaar voor de geselecteerde vlucht.")
else:
    # Bereken de vluchtduur in een leesbaar formaat
    max_time_secs = df['Time (secs)'].max()
    max_time_min = max_time_secs // 60
    max_time_sec = max_time_secs % 60
    flight_duration = f"{max_time_min} minuten en {max_time_sec} seconden"

    # Creëer de folium kaart
    map_center = [df['[3d Latitude]'].mean(), df['[3d Longitude]'].mean()]
    flight_map = folium.Map(location=map_center, zoom_start=6)

    # Voeg markers toe voor de banen
    for baan, coord in banen.items():
        folium.Marker(location=coord, popup=baan, icon=folium.Icon(color='blue')).add_to(flight_map)

    # Definieer de hoogtes en kleuren
    altitude_bins = [-10, 1000, 3000, 7500, 12000]  # Definieer de hoogte-intervallen
    altitude_colors = ['blue', 'green', 'orange', 'red', 'yellow']  # Kleuren voor elke klasse

    # Voeg de route met verschillende kleuren toe
    coords = list(zip(df['[3d Latitude]'], df['[3d Longitude]'], df['[3d Altitude M]']))

    # Voeg de gekleurde lijn toe aan de kaart
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        
        # Bepaal de kleur op basis van de hoogte
        for j in range(len(altitude_bins) - 1):
            if altitude_bins[j] < start[2] <= altitude_bins[j + 1]:
                color = altitude_colors[j]
                break

        # Voeg alleen toe als de coördinaten geldig zijn
        if pd.notna(start[0]) and pd.notna(start[1]) and pd.notna(end[0]) and pd.notna(end[1]):
            folium.PolyLine(locations=[[start[0], start[1]], [end[0], end[1]]], color=color, weight=5).add_to(flight_map)

    # Voeg een legenda toe
    colormap = branca.colormap.LinearColormap(colors=altitude_colors, vmin=min(altitude_bins), vmax=max(altitude_bins))
    colormap.caption = 'Altitude (M)'
    colormap.add_to(flight_map)

    # Render de kaart in Streamlit met vluchtduur
    st.title(f"Flight Route for {selected_vlucht}")
    st.write(f"Vluchtduur: {flight_duration}")  # Voeg de vluchtduur toe
    st_data = st_folium(flight_map, width=725)

# Zet de datum- en tijdstempels om naar datetime-formaat
df3['gepl_aank'] = pd.to_datetime(df3['date'] + ' ' + df3['gepl_aank'], format='%d/%m/%Y %H:%M:%S')
df3['werk_aank'] = pd.to_datetime(df3['date'] + ' ' + df3['werk_aank'], format='%d/%m/%Y %H:%M:%S')

# Bereken de vertraging (in minuten)
df3['vertraging'] = (df3['werk_aank'] - df3['gepl_aank']).dt.total_seconds() / 60  # Omrekeningen naar minuten

# Bekijk het resultaat
print(df3[['FLT', 'gepl_aank', 'werk_aank', 'vertraging']])


'''
Deze code is voor het lineare voorspellings model die de vertraging voorspeld.
'''

