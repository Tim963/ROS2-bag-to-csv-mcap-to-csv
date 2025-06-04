import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from datetime import datetime

# CSV-Datei einlesen
df = pd.read_csv('teko1_odom.csv')  # Pfad anpassen

# Daten extrahieren
x = df['pose_pose_position_x'].values
y = df['pose_pose_position_y'].values
timestamps = df['header_stamp_sec'].values

# UNIX-Zeitstempel in datetime-Objekte konvertieren
times = [datetime.fromtimestamp(ts) for ts in timestamps]

# Relative Zeit in Sekunden seit Start berechnen
time_sec = timestamps - timestamps.min()

# Erstelle Figure und Axes
fig, ax = plt.subplots(figsize=(14, 10))

# 1. Liniensegmente mit Farbverlauf (zeitliche Entwicklung)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# LineCollection für farbige Linien
lc = LineCollection(
    segments,
    cmap='viridis',  # Farbschema: 'jet', 'plasma', 'inferno', 'magma'
    norm=plt.Normalize(time_sec.min(), time_sec.max()),
    linewidth=2.5,
    alpha=0.8
)
lc.set_array(time_sec[:-1])  # Farben basierend auf Zeit zuweisen
ax.add_collection(lc)

# 2. Punkte plotten mit zeitlicher Farbcodierung
scatter = ax.scatter(
    x, y,
    c=time_sec,
    cmap='viridis',
    s=40,           # Punktgröße
    edgecolor='w',   # Weiße Umrandung
    linewidth=0.8,
    zorder=3         # Punkte über Linien zeichnen
)

# Farbbalken hinzufügen und konfigurieren
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label('Zeit seit Start [s]', fontsize=12)

# Zeitinformation für Farbleiste berechnen
time_min = time_sec.min()
time_max = time_sec.max()
duration = time_max - time_min

# Intelligente Tick-Berechnung basierend auf Dauer
if duration <= 10:
    ticks = np.linspace(time_min, time_max, 6)
elif duration <= 60:
    ticks = np.arange(0, np.ceil(time_max)+1, 5)
else:
    ticks = np.linspace(time_min, time_max, 8)

cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

# Plot-Einstellungen
ax.set_title(f'Roboter-Trajektorie: {len(x)} Positionen\n'
            f'Gesamtdauer: {duration:.2f} Sekunden', fontsize=14)
ax.set_xlabel('X-Position [m]', fontsize=12)
ax.set_ylabel('Y-Position [m]', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')  # Gleiche Skalierung beider Achsen

# Start- und Endpunkt markieren mit Zeitinformation
ax.plot(x[0], y[0], 'go', markersize=12, label=f'Start: t={time_sec[0]:.1f}s')
ax.plot(x[-1], y[-1], 'ro', markersize=12, label=f'Ende: t={time_sec[-1]:.1f}s')

# Zeitstempel als Annotation hinzufügen (jede 10. Position)
for i in range(0, len(x), max(1, len(x)//20)):
    if i > 0 and i < len(x)-1:
        ax.annotate(
            f"{time_sec[i]:.1f}s",
            (x[i], y[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8,
            alpha=0.7
        )

# Legende hinzufügen
ax.legend(loc='best', fontsize=10)

# Zusätzliche Zeitinformation anzeigen
time_info = (f"Startzeit: {times[0].strftime('%H:%M:%S')}\n"
             f"Endzeit: {times[-1].strftime('%H:%M:%S')}\n"
             f"Messdauer: {duration:.2f} Sekunden")
ax.text(0.98, 0.02, time_info,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('roboter_trajektorie_zeitcodiert.png', dpi=300)
plt.show()