import folium
from folium.plugins import FloatImage

WARNING_ICON = "https://upload.wikimedia.org/wikipedia/commons/5/50/Yellow_warning_icon.png"

def make_windfarm_map(df):
    m = folium.Map(location=[-26.5, 135], zoom_start=4)

    # Add OpenWeather wind tile layer
    folium.TileLayer(
        tiles="https://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?appid=f05a24cb7f6de532c1dfbe891c86552e",
        attr="OpenWeatherMap",
        name="Wind Layer",
        opacity=0.6
    ).add_to(m)

    for _, row in df.iterrows():
        lat, lon = row["Lat"], row["Lon"]
        if lat is None or lon is None:
            continue

        actual = row["Actual (MW)"]
        expected = row["Expected (MW)"]
        cap = row["Capacity (MW)"]
        error = row["Error (%)"]

        scale = 500  # ring scaling
        outer_radius = cap * scale if cap else 0
        inner_radius = actual * scale if actual else 0

        # Determine colour
        if error and error > 20:
            color = "yellow"
        else:
            color = "green"

        # Outer ring (capacity)
        folium.Circle(
            location=[lat, lon],
            radius=outer_radius,
            color="black",
            weight=1,
            fill=False
        ).add_to(m)

        # Inner ring (actual)
        folium.Circle(
            location=[lat, lon],
            radius=inner_radius,
            color=color,
            fill=True,
            fill_opacity=0.5
        ).add_to(m)

        # Warning icon
        if error and error > 20:
            folium.Marker(
                [lat + 0.1, lon + 0.1],
                icon=folium.CustomIcon(WARNING_ICON, icon_size=(30, 30))
            ).add_to(m)

        popup = f"""
            <b>{row['Name']}</b><br>
            Participant: {row['Participant']}<br>
            Capacity: {row['Capacity (MW)']} MW<br>
            Actual: {actual} MW<br>
            Expected: {expected} MW<br>
            Error: {error:.1f}%<br>
        """


        folium.Marker([lat, lon], popup=popup).add_to(m)

    folium.LayerControl().add_to(m)

    return m._repr_html_()