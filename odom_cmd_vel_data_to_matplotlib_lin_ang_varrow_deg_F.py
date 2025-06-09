import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from datetime import datetime
from math import sin, cos, pi

# Beide CSV-Dateien einlesen
df_odom = pd.read_csv('odom.csv')  # Pfad anpassen
df_cmd = pd.read_csv('cmd_vel.csv')  # Pfad anpassen

# Odom-Daten extrahieren
x = df_odom['pose_pose_position_x'].values
y = df_odom['pose_pose_position_y'].values
odom_timestamps = df_odom['header_stamp_sec'].values

# Orientierungsdaten (Quaternion) extrahieren
qx = df_odom['pose_pose_orientation_x'].values
qy = df_odom['pose_pose_orientation_y'].values
qz = df_odom['pose_pose_orientation_z'].values
qw = df_odom['pose_pose_orientation_w'].values

odom_linear_vel = df_odom['twist_twist_linear_x'].values
odom_angular_vel = df_odom['twist_twist_angular_z'].values

# Cmd_vel-Daten extrahieren
cmd_timestamps = df_cmd['header_stamp_sec'].values
cmd_linear_vel = df_cmd['twist_linear_x'].values
cmd_angular_vel = df_cmd['twist_angular_z'].values

# Quaternion zu Euler-Winkel (nur Yaw/Winkel um Z-Achse)
def quaternion_to_yaw(qx, qy, qz, qw):
    yaw = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    return yaw

# Yaw-Winkel für alle Orientierungen berechnen (in rad)
yaw = quaternion_to_yaw(qx, qy, qz, qw)

# Aktive Steuerbefehle identifizieren
active_mask = (np.abs(cmd_linear_vel) > 0) | (np.abs(cmd_angular_vel) > 0)

# Zeitpunkt des ersten Steuersignals finden
first_active_index = np.argmax(active_mask)
t_start = cmd_timestamps[first_active_index]

# Relative Zeit berechnen (0 beim ersten Steuersignal)
time_sec = odom_timestamps - t_start
cmd_time_sec = cmd_timestamps - t_start

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
yaw_active = yaw[valid_indices]

# Figure mit drei Subplots erstellen
fig, axs = plt.subplots(3, 1, figsize=(14, 16), 
                        gridspec_kw={'height_ratios': [3, 1, 1]},
                        constrained_layout=True)
ax_traj, ax_lin, ax_ang = axs

# 1. Subplot: Trajektorie mit Farbverlauf (Zeit)
# ==============================================
points = np.array([x_active, y_active]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Farbnormalisierung NUR für den aktiven Zeitbereich
norm = plt.Normalize(time_sec_active.min(), time_sec_active.max())

lc = LineCollection(
    segments,
    cmap='viridis',
    norm=norm,
    linewidth=2.5,
    alpha=0.8
)
lc.set_array(time_sec_active[:-1])
ax_traj.add_collection(lc)

# Punkte plotten mit gleicher Farbnormalisierung
scatter = ax_traj.scatter(
    x_active, y_active,
    c=time_sec_active,
    cmap='viridis',
    norm=norm,
    s=40,
    edgecolor='w',
    linewidth=0.8,
    zorder=3
)

# Farbbalken für Zeit mit korrekter Skalierung
cbar = fig.colorbar(scatter, ax=ax_traj, pad=0.01, aspect=10, shrink=0.435)
cbar.set_label('Zeit seit erstem Steuersignal [s]', fontsize=12)

# Pfeile für Anfangs- und Endausrichtung mit Gradangaben
traj_scale = np.max([np.ptp(x_active), np.ptp(y_active)])
arrow_length = 0.1
text_offset = 0.02

# Anfangsausrichtung
start_yaw_deg = np.degrees(yaw_active[0]) % 360
start_arrow_dx = arrow_length * cos(yaw_active[0])
start_arrow_dy = arrow_length * sin(yaw_active[0])
ax_traj.arrow(x_active[0], y_active[0], 
              start_arrow_dx, start_arrow_dy, 
              head_width=arrow_length*0.25, 
              head_length=arrow_length*0.3,
              fc='green', ec='green', 
              width=arrow_length*0.05, 
              zorder=4)

# Text für Startausrichtung
ax_traj.text(x_active[0] + start_arrow_dx + text_offset, 
             y_active[0] + start_arrow_dy + text_offset,
             f'φ={start_yaw_deg:.1f}°\n t={time_sec_active[0]:.1f}s',
             color='green', fontsize=8, weight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Endausrichtung
end_yaw_deg = np.degrees(yaw_active[-1]) % 360
end_arrow_dx = arrow_length * cos(yaw_active[-1])
end_arrow_dy = arrow_length * sin(yaw_active[-1])
ax_traj.arrow(x_active[-1], y_active[-1], 
              end_arrow_dx, end_arrow_dy, 
              head_width=arrow_length*0.25,
              head_length=arrow_length*0.3,
              fc='red', ec='red', 
              width=arrow_length*0.05,
              zorder=4)

# Text für Endausrichtung
ax_traj.text(x_active[-1] + end_arrow_dx + text_offset, 
             y_active[-1] + end_arrow_dy + text_offset,
             f'φ={end_yaw_deg:.1f}°\n t={time_sec_active[-1]:.1f}s',
             color='red', fontsize=8, weight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Plot-Einstellungen
duration = time_sec_active.max() - time_sec_active.min()
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

ax_ang.set_xlabel('Zeit seit erstem Steuersignal [s]', fontsize=12)
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
plt.savefig('roboter_trajektorie_analyse.png', dpi=300, bbox_inches='tight')
plt.show()
