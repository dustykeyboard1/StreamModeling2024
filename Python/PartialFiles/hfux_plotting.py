import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#2D Plot of stream temperature

# Create a figure
fig, ax = plt.subplots()

# ax.imshow() - https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
cax = ax.imshow(T, aspect='auto', cmap='jet', origin='lower',
                extent=[np.min(time_mod), np.max(time_mod), np.min(dist_mod), np.max(dist_mod)])

# Add a colorbar with label
# plt.colorbar() - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
cbar = plt.colorbar(cax)
cbar.set_label('Temperature (°C)', fontsize=12, fontweight='bold')

# Set title and labels
plot_title = 'Modeled Stream Temperature'
xlab = 'Time (min)'
ylab = 'Distance (m)'
ax.set_title(plot_title, fontsize=12, fontweight='bold')
ax.set_xlabel(xlab, fontsize=11)
ax.set_ylabel(ylab, fontsize=11)

plt.show(block = False)

#3D plot of stream temperature
fig = plt.figure()

# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot
# ax.plot_surface() - https://matplotlib.org/stable/plot_types/3D/surface3d_simple.html#plot-surface-x-y-z
surface = ax.plot_surface(time_mod, dist_mod, T, cmap='jet')

# Add a colorbar with label
cbar = fig.colorbar(surface)
cbar.set_label('Temperature (°C)', fontsize=11, fontweight='bold')

# Set title and labels
plot_title = 'Modeled Stream Temperature'
xlab = 'Time (min)'
ylab = 'Distance (m)'
zlab = 'Temp (°C)'
ax.set_title(plot_title, fontsize=12, fontweight='bold')
ax.set_xlabel(xlab, fontsize=11)
ax.set_ylabel(ylab, fontsize=11)
ax.set_zlabel(zlab, fontsize=11)

plt.show(block = False)

#Plot of heat fluxes
fig = plt.figure()

# Subplot 1
plt.subplot(3, 2, 1)
plt.plot(time_mod, np.mean(heat_flux / 60, axis=0), 'k')
plt.title('Total Heat Flux', fontweight='bold')
plt.ylabel('Energy Flux (W/m^2)', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(heat_flux / 60, axis=0)), np.max(np.mean(heat_flux / 60, axis=0))])

# Subplot 2
plt.subplot(3, 2, 2)
plt.plot(time_mod, np.mean(shortwave, axis=0), 'r')
plt.title('Shortwave Radiation', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(shortwave, axis=0)), np.max(np.mean(shortwave, axis=0))])

# Subplot 3
plt.subplot(3, 2, 3)
plt.plot(time_mod, np.mean(longwave, axis=0), 'b')
plt.title('Longwave Radiation', fontweight='bold')
plt.ylabel('Energy Flux (W/m^2)', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(longwave, axis=0)), np.max(np.mean(longwave, axis=0))])

# Subplot 4
plt.subplot(3, 2, 4)
plt.plot(time_mod, np.mean(latent, axis=0), 'g')
plt.title('Latent Heat Flux', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(latent, axis=0)), np.max(np.mean(latent, axis=0))])

# Subplot 5
plt.subplot(3, 2, 5)
plt.plot(time_mod, np.mean(bed, axis=0), 'm')
plt.title('Bed Conduction', fontweight='bold')
plt.xlabel('Time (min)', fontweight='bold')
plt.ylabel('Energy Flux (W/m^2)', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(bed, axis=0)), np.max(np.mean(bed, axis=0))])

# Subplot 6
plt.subplot(3, 2, 6)
plt.plot(time_mod, np.mean(sensible, axis=0), 'y')
plt.title('Sensible Heat Flux', fontweight='bold')
plt.xlabel('Time (min)', fontweight='bold')
plt.axis([np.min(time_mod), np.max(time_mod), np.min(np.mean(sensible, axis=0)), np.max(np.mean(sensible, axis=0))])

plt.show(block = False)

#Plot of heat fluxes: comparison 
plt.figure()

# Plot data
plt.plot(time_mod, np.mean(heat_flux / 60, axis=0), 'k', label='Total Heat Flux')
plt.plot(time_mod, np.mean(shortwave, axis=0), 'r', label='Solar Radiation')
plt.plot(time_mod, np.mean(longwave, axis=0), 'b', label='Longwave Radiation')
plt.plot(time_mod, np.mean(latent, axis=0), 'g', label='Latent Heat Flux')
plt.plot(time_mod, np.mean(bed, axis=0), 'c', label='Streambed Conduction')
plt.plot(time_mod, np.mean(sensible, axis=0), 'm', label='Sensible Heat Flux')

# Set axis properties
plt.axis('normal')
plt.xlim([np.min(time_mod), np.max(time_mod)])

# Add title and labels
plt.title('Energy Fluxes', fontsize=12, fontweight='bold')
plt.xlabel('Time (min)', fontsize=11, fontweight='bold')
plt.ylabel('Energy Flux (W/m^2)', fontsize=11, fontweight='bold')

# Add legend
plt.legend(loc='best')

plt.show(block = False)