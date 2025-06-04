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
linear_vel = df['twist_twist_linear_x'].values
angular_vel = df['twist_twist_angular_z'].values

# UNIX-Zeitstempel in datetime-Objekte konvertieren
times = [datetime.fromtimestamp(ts) for ts in timestamps]

# Relative Zeit in Sekunden seit Start berechnen
time_sec = timestamps - timestamps.min()
duration = time_sec.max() - time_sec.min()

# Figure mit drei Subplots erstellen
fig, axs = plt.subplots(3, 1, figsize=(14, 16), 
                        gridspec_kw={'height_ratios': [3, 1, 1]},
                        constrained_layout=True)
ax_traj, ax_lin, ax_ang = axs

# 1. Subplot: Trajektorie mit Farbverlauf (Zeit)
# ==============================================
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap='viridis',
    norm=plt.Normalize(time_sec.min(), time_sec.max()),
    linewidth=2.5,
    alpha=0.8
)
lc.set_array(time_sec[:-1])
ax_traj.add_collection(lc)

# Punkte plotten
scatter = ax_traj.scatter(
    x, y,
    c=time_sec,
    cmap='viridis',
    s=40,
    edgecolor='w',
    linewidth=0.8,
    zorder=3
)

# Farbbalken für Zeit
cbar = fig.colorbar(scatter, ax=ax_traj, pad=0.01)
cbar.set_label('Zeit seit Start [s]', fontsize=12)

# Start- und Endpunkt markieren
ax_traj.plot(x[0], y[0], 'go', markersize=12, label=f'Start: t={time_sec[0]:.1f}s')
ax_traj.plot(x[-1], y[-1], 'ro', markersize=12, label=f'Ende: t={time_sec[-1]:.1f}s')
ax_traj.legend(loc='best', fontsize=10)

# Plot-Einstellungen
ax_traj.set_title(f'Roboter-Trajektorie ({len(x)} Positionen, Dauer: {duration:.2f}s)', fontsize=14)
ax_traj.set_xlabel('X-Position [m]', fontsize=12)
ax_traj.set_ylabel('Y-Position [m]', fontsize=12)
ax_traj.grid(True, linestyle='--', alpha=0.5)
ax_traj.set_aspect('equal')

# 2. Subplot: Lineare Geschwindigkeit
# ===================================
ax_lin.plot(time_sec, linear_vel, 
            color='royalblue', 
            linewidth=2.0,
            label='Lineare Geschwindigkeit')

# Füllung unter der Kurve
ax_lin.fill_between(time_sec, linear_vel, 0,
                    where=(linear_vel >= 0),
                    color='skyblue',
                    alpha=0.5,
                    interpolate=True)

ax_lin.fill_between(time_sec, linear_vel, 0,
                    where=(linear_vel < 0),
                    color='salmon',
                    alpha=0.5,
                    interpolate=True)

# Null-Linie und Max-Werte
ax_lin.axhline(0, color='k', linestyle='-', alpha=0.3)
max_lin = np.max(np.abs(linear_vel)) * 1.2
ax_lin.set_ylim(-max_lin, max_lin)

ax_lin.set_ylabel('Geschwindigkeit [m/s]', fontsize=12)
ax_lin.grid(True, linestyle='--', alpha=0.5)
ax_lin.legend(loc='upper right', fontsize=10)
ax_lin.set_title('Lineare Geschwindigkeit (X-Richtung)', fontsize=12)

# 3. Subplot: Winkelgeschwindigkeit
# =================================
ax_ang.plot(time_sec, angular_vel, 
            color='darkorange', 
            linewidth=2.0,
            label='Winkelgeschwindigkeit')

# Füllung unter der Kurve
ax_ang.fill_between(time_sec, angular_vel, 0,
                    where=(angular_vel >= 0),
                    color='gold',
                    alpha=0.5,
                    interpolate=True)

ax_ang.fill_between(time_sec, angular_vel, 0,
                    where=(angular_vel < 0),
                    color='darkorange',
                    alpha=0.3,
                    interpolate=True)

# Null-Linie und Max-Werte
ax_ang.axhline(0, color='k', linestyle='-', alpha=0.3)
max_ang = np.max(np.abs(angular_vel)) * 1.2
ax_ang.set_ylim(-max_ang, max_ang)

ax_ang.set_xlabel('Zeit seit Start [s]', fontsize=12)
ax_ang.set_ylabel('Drehrate [rad/s]', fontsize=12)
ax_ang.grid(True, linestyle='--', alpha=0.5)
ax_ang.legend(loc='upper right', fontsize=10)
ax_ang.set_title('Winkelgeschwindigkeit (Z-Achse)', fontsize=12)

# Gemeinsame X-Achse einrichten
ax_lin.sharex(ax_ang)
ax_lin.tick_params(labelbottom=False)
ax_ang.set_xlim(time_sec.min(), time_sec.max())

# Gesamttitel
fig.suptitle('Roboter-Trajektorie und Geschwindigkeitsanalyse', 
             fontsize=16, 
             y=0.98)

# Speichern und anzeigen
plt.savefig('roboter_trajektorie_mit_geschwindigkeiten.png', dpi=300, bbox_inches='tight')
plt.show()
