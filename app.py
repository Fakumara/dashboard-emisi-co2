import streamlit as st
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# Load dataset
file_path = 'id.csv'
data = pd.read_csv(file_path)

# Haversine Function for Distance Calculation
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Emission calculation based on GHG Protocol
def calculate_emissions(distance, emission_factor, scope):
    if scope == 1:
        return distance * emission_factor  # Direct emissions (e.g., fuel combustion)
    elif scope == 2:
        return 0  # No electricity-based emissions for transportation
    elif scope == 3:
        return distance * emission_factor * 0.8  # Adjusted factor for third-party transport
    else:
        raise ValueError("Invalid scope. Choose 1, 2, or 3.")

# Streamlit App
st.set_page_config(page_title="Dashboard Emisi CO2", layout="wide")

# Sidebar
st.sidebar.title("CO2 Summary")

# Dropdown to select the center location
city_names = data['city'].unique()
selected_city = st.sidebar.selectbox("Pilih Pusat Distribusi:", city_names)

# Get center location details
center_location = data[data['city'] == selected_city].iloc[0]
center_lat, center_lng = center_location['lat'], center_location['lng']

# Calculate total emissions and other metrics
total_emissions_scope1 = data.apply(lambda row: calculate_emissions(haversine(center_lat, center_lng, row['lat'], row['lng']), 1.34, scope=1), axis=1).sum()
total_emissions_scope3 = data.apply(lambda row: calculate_emissions(haversine(center_lat, center_lng, row['lat'], row['lng']), 1.34, scope=3), axis=1).sum()
total_emissions = total_emissions_scope1 + total_emissions_scope3

# Add styled metrics to the sidebar
st.sidebar.markdown(
    f"""
    <div style="padding: 15px; border-radius: 5px; background-color: #FFD700; color: #000; text-align: center;">
        <h3>Total carbon footprint</h3>
        <h1 style="font-size: 36px;">{total_emissions:,.2f} kCO₂e</h1>
        <p>Equivalent to {total_emissions * 33.3:.0f} Mangrove trees in 8 years of lifespan</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    f"""
    <div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <h4>Emission scope 1</h4>
        <h2>{total_emissions_scope1:,.2f} kCO₂e</h2>
        <p>Emisi langsung</p>
    </div>
    <div style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px;">
        <h4>Emission scope 3</h4>
        <h2>{total_emissions_scope3:,.2f} kCO₂e</h2>
        <p>Emisi tidak langsung (transportasi pihak ketiga).</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Emission factor (1.34 kg CO2 per km)
emission_factor_per_km = 1.34

# Title and Description
st.title("Dashboard Analisis Emisi CO2")
st.markdown("""
<div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; font-family: Arial, sans-serif;">
    <p>Dashboard ini menampilkan analisis emisi CO2 dari berbagai lokasi berdasarkan jarak dari kota pusat.</p>
    <ul>
        <li><strong>Scope 1</strong>: Emisi langsung.</li>
        <li><strong>Scope 3</strong>: Emisi tidak langsung (transportasi pihak ketiga).</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Create the map
st.header("Peta Distribusi dan Emisi CO2")
col1, col2 = st.columns([2, 1])
with col1:
    m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
    folium.Marker(
        location=[center_lat, center_lng],
        popup=f"<b>{selected_city}</b><br>Lokasi pusat",
        tooltip=f"Center: {selected_city}",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    marker_cluster = MarkerCluster().add_to(m)

    # Initialize lists for rankings
    ranking_data = []

    # Iterate over dataset to add markers and calculate emissions
    for _, row in data.iterrows():
        city = row['city']
        lat = row['lat']
        lng = row['lng']
        distance = haversine(center_lat, center_lng, lat, lng)
        emissions_scope1 = calculate_emissions(distance, emission_factor_per_km, scope=1)
        emissions_scope3 = calculate_emissions(distance, emission_factor_per_km, scope=3)

        ranking_data.append({
            'City': city,
            'Distance': distance,
            'Scope 1 Emissions': emissions_scope1,
            'Scope 3 Emissions': emissions_scope3
        })

        folium.Marker(
            location=[lat, lng],
            popup=(
                f"<b>Distance:</b> {distance:.2f} km<br>"
                f"<b>Scope 1 Emissions:</b> {emissions_scope1:.2f} kg CO2<br>"
                f"<b>Scope 3 Emissions:</b> {emissions_scope3:.2f} kg CO2"
            ),
            tooltip=f"{city}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(marker_cluster)
        folium.PolyLine(
            locations=[[center_lat, center_lng], [lat, lng]],
            color='green',
            weight=2
        ).add_to(m)

    st_folium(m, width=700, height=500)

# Convert ranking data to DataFrame
ranking_df = pd.DataFrame(ranking_data)

# Display Table
with col2:
    st.header("Tabel Hasil Kalkulasi Emisi CO2")
    st.dataframe(ranking_df)

# Visualizations
st.header("Visualisasi Data")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Ranking Berdasarkan Jarak dan Emisi CO2")
    selected_metric = st.selectbox("Pilih Metode Ranking:", ["Distance", "Scope 1 Emissions", "Scope 3 Emissions"], key="ranking")
    ranking_df = ranking_df.sort_values(by=selected_metric, ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ranking_df['City'], ranking_df[selected_metric], color='skyblue')
    ax.set_title(f"Ranking Berdasarkan {selected_metric}", fontsize=14)
    ax.set_xlabel("City", fontsize=10)
    ax.set_ylabel(selected_metric, fontsize=10)
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

with col4:
    st.subheader("Distribusi Emisi CO2 (Scope 1 dan Scope 3)")
    total_emissions_scope1 = ranking_df['Scope 1 Emissions'].sum()
    total_emissions_scope3 = ranking_df['Scope 3 Emissions'].sum()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        [total_emissions_scope1, total_emissions_scope3],
        labels=["Scope 1", "Scope 3"],
        autopct='%1.1f%%',
        startangle=90,
        colors=["#ff9999", "#66b3ff"]
    )
    ax2.set_title("Distribusi Emisi CO2")
    st.pyplot(fig2)

# Line Chart
st.subheader("Analisis Jarak dan Emisi CO2")
fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(ranking_df['City'], ranking_df['Distance'], marker='o', label='Distance', color='blue')
ax3.plot(ranking_df['City'], ranking_df['Scope 1 Emissions'], marker='o', label='Scope 1 Emissions', color='red')
ax3.plot(ranking_df['City'], ranking_df['Scope 3 Emissions'], marker='o', label='Scope 3 Emissions', color='green')
ax3.set_title("Analisis Jarak dan Emisi CO2", fontsize=14)
ax3.set_xlabel("City", fontsize=10)
ax3.set_ylabel("Values", fontsize=10)
ax3.tick_params(axis='x', rotation=90)
ax3.legend()
st.pyplot(fig3)
