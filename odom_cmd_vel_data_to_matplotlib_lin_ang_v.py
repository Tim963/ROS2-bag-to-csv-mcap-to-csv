import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.dates as mdates
from datetime import datetime

# Beide CSV-Dateien einlesen
df_odom = pd.read_csv('odom.csv')  # Pfad anpassen
df_cmd = pd.read_csv('cmd_vel.csv')  # Pfad anpassen

# Odom-Daten extrahieren
x = df_odom['pose_pose_position_x'].values
y = df_odom['pose_pose_position_y'].values
odom_timestamps = df_odom['header_stamp_sec'].values
odom_linear_vel = df_odom['twist_twist_linear_x'].values
odom_angular_vel = df_odom['twist_twist_angular_z'].values

# Cmd_vel-Daten extrahieren
cmd_timestamps = df_cmd['header_stamp_sec'].values
cmd_linear_vel = df_cmd['twist_linear_x'].values
cmd_angular_vel = df_cmd['twist_angular_z'].values

# UNIX-Zeitstempel in datetime-Objekte konvertieren
odom_times = [datetime.fromtimestamp(ts) for ts in odom_timestamps]
cmd_times = [datetime.fromtimestamp(ts) for ts in cmd_timestamps]

# Relative Zeit in Sekunden seit Start berechnen (basierend auf Odom)
time_sec = odom_timestamps - odom_timestamps.min()
cmd_time_sec = cmd_timestamps - odom_timestamps.min()

# Aktive Steuerbefehle identifizieren
active_mask = (np.abs(cmd_linear_vel) > 0) | (np.abs(cmd_angular_vel) > 0)

# Zeitbereiche mit aktiven Steuerbefehlen finden
active_periods = []
current_start = None
for i, is_active in enumerate(active_mask):
    if is_active and current_start is None:
        current_start = cmd_time_sec[i]
    elif not is_active and current_start is not None:
        active_periods.append((current_start, cmd_time_sec[i]))
        current_start = None
if current_start is not None:
    active_periods.append((current_start, cmd_time_sec[-1]))

# Indizes der Odom-Daten finden, die in aktiven Perioden liegen
valid_indices = np.zeros_like(odom_timestamps, dtype=bool)
for start, end in active_periods:
    valid_indices |= (time_sec >= start) & (time_sec <= end)

# Nur Daten in aktiven Perioden behalten
x_active = x[valid_indices]
y_active = y[valid_indices]
time_sec_active = time_sec[valid_indices]
odom_linear_vel_active = odom_linear_vel[valid_indices]
odom_angular_vel_active = odom_angular_vel[valid_indices]

# Gesamtdauer berechnen
duration = time_sec_active.max() - time_sec_active.min()

# Figure mit drei Subplots erstellen
fig, axs = plt.subplots(3, 1, figsize=(14, 16), 
                        gridspec_kw={'height_ratios': [3, 1, 1]},
                        constrained_layout=True)
ax_traj, ax_lin, ax_ang = axs

# 1. Subplot: Trajektorie mit Farbverlauf (Zeit)
# ==============================================
points = np.array([x_active, y_active]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

lc = LineCollection(
    segments,
    cmap='viridis',
    norm=plt.Normalize(time_sec_active.min(), time_sec_active.max()),
    linewidth=2.5,
    alpha=0.8
)
lc.set_array(time_sec_active[:-1])
ax_traj.add_collection(lc)

# Punkte plotten
scatter = ax_traj.scatter(
    x_active, y_active,
    c=time_sec_active,
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
ax_traj.plot(x_active[0], y_active[0], 'go', markersize=12, label=f'Start: t={time_sec_active[0]:.1f}s')
ax_traj.plot(x_active[-1], y_active[-1], 'ro', markersize=12, label=f'Ende: t={time_sec_active[-1]:.1f}s')
ax_traj.legend(loc='best', fontsize=10)

# Plot-Einstellungen
ax_traj.set_title(f'Roboter-Trajektorie während aktiver Steuerung ({len(x_active)} Positionen, Dauer: {duration:.2f}s)', fontsize=14)
ax_traj.set_xlabel('X-Position [m]', fontsize=12)
ax_traj.set_ylabel('Y-Position [m]', fontsize=12)
ax_traj.grid(True, linestyle='--', alpha=0.5)
ax_traj.set_aspect('equal')

# 2. Subplot: Lineare Geschwindigkeit (Odom und Cmd)
# ================================================
ax_lin.plot(time_sec_active, odom_linear_vel_active, 
            color='royalblue', 
            linewidth=2.0,
            label='Gemessene Geschwindigkeit (Odom)')

ax_lin.plot(cmd_time_sec, cmd_linear_vel, 
            color='darkgreen', 
            linewidth=1.5,
            linestyle='--',
            alpha=0.7,
            label='Befehlsgeschwindigkeit (Cmd_vel)')

# Füllung unter der Kurve
ax_lin.fill_between(time_sec_active, odom_linear_vel_active, 0,
                    where=(odom_linear_vel_active >= 0),
                    color='skyblue',
                    alpha=0.5,
                    interpolate=True)

ax_lin.fill_between(time_sec_active, odom_linear_vel_active, 0,
                    where=(odom_linear_vel_active < 0),
                    color='salmon',
                    alpha=0.5,
                    interpolate=True)

# Null-Linie und Max-Werte
ax_lin.axhline(0, color='k', linestyle='-', alpha=0.3)
max_lin = max(np.max(np.abs(odom_linear_vel_active)), np.max(np.abs(cmd_linear_vel))) * 1.2
ax_lin.set_ylim(-max_lin, max_lin)

ax_lin.set_ylabel('Geschwindigkeit [m/s]', fontsize=12)
ax_lin.grid(True, linestyle='--', alpha=0.5)
ax_lin.legend(loc='upper right', fontsize=10)
ax_lin.set_title('Lineare Geschwindigkeit (X-Richtung)', fontsize=12)

# 3. Subplot: Winkelgeschwindigkeit (Odom und Cmd)
# ================================================
ax_ang.plot(time_sec_active, odom_angular_vel_active, 
            color='darkorange', 
            linewidth=2.0,
            label='Gemessene Drehrate (Odom)')

ax_ang.plot(cmd_time_sec, cmd_angular_vel, 
            color='purple', 
            linewidth=1.5,
            linestyle='--',
            alpha=0.7,
            label='Befehlsdrehrate (Cmd_vel)')

# Füllung unter der Kurve
ax_ang.fill_between(time_sec_active, odom_angular_vel_active, 0,
                    where=(odom_angular_vel_active >= 0),
                    color='gold',
                    alpha=0.5,
                    interpolate=True)

ax_ang.fill_between(time_sec_active, odom_angular_vel_active, 0,
                    where=(odom_angular_vel_active < 0),
                    color='darkorange',
                    alpha=0.3,
                    interpolate=True)

# Null-Linie und Max-Werte
ax_ang.axhline(0, color='k', linestyle='-', alpha=0.3)
max_ang = max(np.max(np.abs(odom_angular_vel_active)), np.max(np.abs(cmd_angular_vel))) * 1.2
ax_ang.set_ylim(-max_ang, max_ang)

ax_ang.set_xlabel('Zeit seit Start [s]', fontsize=12)
ax_ang.set_ylabel('Drehrate [rad/s]', fontsize=12)
ax_ang.grid(True, linestyle='--', alpha=0.5)
ax_ang.legend(loc='upper right', fontsize=10)
ax_ang.set_title('Winkelgeschwindigkeit (Z-Achse)', fontsize=12)

# Gemeinsame X-Achse einrichten
ax_lin.sharex(ax_ang)
ax_lin.tick_params(labelbottom=False)
ax_ang.set_xlim(time_sec_active.min(), time_sec_active.max())

# Gesamttitel
fig.suptitle('Roboter-Trajektorie und Geschwindigkeitsanalyse (nur aktive Steuerperioden)', 
             fontsize=16, 
             y=0.98)

# Speichern und anzeigen
plt.savefig('roboter_trajektorie_mit_geschwindigkeiten_aktiv.png', dpi=300, bbox_inches='tight')
plt.show()
