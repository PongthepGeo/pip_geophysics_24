#-----------------------------------------------------------------------------------------#
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
#-----------------------------------------------------------------------------------------#
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

def simple_harmonic_plot(distance, cosine_values, sine_values):
    plt.plot(distance, cosine_values)
    plt.plot(distance, sine_values)
    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.legend(['Cosine', 'Sine'], loc='upper right')
    plt.title('Cosine and Sine Waves')
    plt.savefig('figure_out/' + 'simple_har.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()

#-----------------------------------------------------------------------------------------#

def simple_harmonic_plot(distance, cosine_values, sine_values):
    plt.plot(distance, cosine_values)
    plt.plot(distance, sine_values)
    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.legend(['Cosine', 'Sine'], loc='upper right')
    plt.title('Cosine and Sine Waves')
    plt.savefig('figure_out/' + 'simple_har.png', format='png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()

#-----------------------------------------------------------------------------------------#
def interference(distance, cosine_values, sine_values):
    plt.plot(distance, cosine_values, linestyle='--', linewidth=0.5)
    plt.plot(distance, sine_values, linestyle='--', linewidth=0.5)
    plt.plot(distance, (cosine_values - sine_values))
    plt.xlabel('Distance (m)')
    plt.ylabel('Amplitude')
    plt.legend(['Cosine', 'Sine', 'New Wave'], loc='upper left')
    plt.title('Interference')
    plt.savefig('figure_out/' + 'interference.png', format='png', bbox_inches='tight',
                transparent=True, pad_inches=0.1)
    plt.show()

#-----------------------------------------------------------------------------------------#

def plot_resolution(image, blur):
    ROW, COL = image.shape
    fig = plt.figure(figsize=(10, 10))  # Adjusted figure size for 2x2 grid
    
    # Top-left: Original image with y-axis visible
    ax1 = fig.add_subplot(2, 2, 1)
    x = [205, 205]
    y = [0, ROW]
    ax1.plot(x, y, color='red', linewidth=1, linestyle='--')
    ax1.imshow(image, aspect='auto')
    ax1.title.set_text('Image')

    # Top-right: Signal at column 205 with y-axis hidden
    ax2 = fig.add_subplot(2, 2, 2)
    axis_y = np.linspace(0, ROW, ROW)
    ax2.plot(image[:, 205], axis_y, color='red', linewidth=1, linestyle='-')
    ax2.title.set_text('Trace 205')
    ax2.invert_yaxis()
    ax2.set_xlabel('Amplitude')
    ax2.set_ylabel('')
    ax2.yaxis.set_visible(False)

    # Bottom-left: Blurred image with y-axis hidden
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x, y, color='pink', linewidth=1, linestyle='--')
    ax3.imshow(blur, aspect='auto')
    ax3.title.set_text('Blur')
    ax3.set_ylabel('')

    # Bottom-right: Signal at column 205 for blurred image with y-axis hidden
    ax4 = fig.add_subplot(2, 2, 4)
    axis_y = np.linspace(0, ROW, ROW)
    ax4.plot(blur[:, 205], axis_y, color='pink', linewidth=1, linestyle='-')
    ax4.title.set_text('Trace 205')
    ax4.invert_yaxis()
    ax4.set_xlabel('Amplitude')
    ax4.set_ylabel('')
    ax4.yaxis.set_visible(False)
    
    plt.savefig('figure_out/resolution.png', format='png', bbox_inches='tight',
                transparent=True, pad_inches=0.1)
    plt.show()

#-----------------------------------------------------------------------------------------#

def gravitational_field_magnitude(x, y, G, M):
    # Calculate distance squared from the center of the Earth
    r_squared = (x - 0)**2 + (y - 0)**2
    # Calculate magnitude of the gravitational field
    magnitude = G * M / r_squared
    return np.sqrt(r_squared), magnitude

#-----------------------------------------------------------------------------------------#

def plot_gravitational_field(radii, magnitudes, colors, labels):
    fig, ax = plt.subplots()
    for i, (radius, magnitude) in enumerate(zip(radii, magnitudes)):
        ax.scatter(radius, magnitude, color=colors[i], s=50, label=labels[i], edgecolors='black')
    ax.legend()
    ax.set_xlabel('Radius (m)')
    ax.set_xscale('log')  # Set x-axis to logarithmic scale
    ax.set_ylabel(r'Gravitational Field Magnitude (m $\cdot$ s$^{-2}$)')
    ax.grid(True, linestyle='--')
    plt.savefig('figure_out/mag_gra.svg', format='svg', bbox_inches='tight',
                transparent=True, pad_inches=0.0)
    plt.show()

#-----------------------------------------------------------------------------------------#

def gravitational_field_magnitude_2(rho, R, x_obs, y_obs, x_sphere, y_sphere, G):
    M = rho * (4 / 3) * np.pi * R**3  # Mass of the sphere
    r_squared = (x_obs - x_sphere)**2 + (y_obs - y_sphere)**2  # Distance squared
    g_magnitude = G * M / r_squared
    return g_magnitude

def plot_gravitational_field(observation_points, observed_g, title, label, color, output_filename, output_dir='figure_out'):
    plt.figure()
    x_observed = np.array([pt[0] for pt in observation_points])
    plt.scatter(x_observed, observed_g, color=color, label=label, s=20, edgecolors='black')
    plt.xlabel('Observation Point X (m)')
    plt.ylabel(r'$|\mathbf{g}|$')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(f'{output_dir}/{output_filename}.svg', format='svg',
                bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

def plot_sphere(sphere_name, sphere, observation_points, observed_g, output_dir='figure_out'):
    title = f'Gravitational Field Magnitude due to {sphere_name}'
    label = sphere_name
    color = sphere['color']
    output_filename = f'sphere_{sphere_name}'
    plot_gravitational_field(observation_points, observed_g, title, label, color, output_filename, output_dir)

def plot_total_and_individual_fields(spheres, observation_points, total_g, individual_g, output_dir='figure_out'):
    plt.figure()
    x_observed = np.array([pt[0] for pt in observation_points])
    
    for sphere_name, observed_g in individual_g.items():
        plt.scatter(x_observed, observed_g, color=spheres[sphere_name]['color'], label=sphere_name, s=20, edgecolors='black')
    
    plt.scatter(x_observed, total_g, marker='x', color='cyan', label='Total Gravity', s=20)
    plt.xlabel('Observation Point X (m)')
    plt.ylabel(r'$|\mathbf{g}|$')
    plt.title('Total Gravitational Field Magnitude and Individual Contributions')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(f'{output_dir}/total_and_individual_fields.svg', format='svg',
                bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

#-----------------------------------------------------------------------------------------#

def plot_density(img_array, density_map, output_dir='figure_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.1]})
    axs[0].imshow(img_array)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    im = axs[1].imshow(density_map, cmap='gray')
    axs[1].set_title('Density Map')
    axs[1].axis('off')
    cbar = plt.colorbar(im, ax=axs[1], fraction=0.038, pad=0.04)
    cbar.set_label('Density')
    plt.savefig(f'{output_dir}/density.svg', format='svg',
                bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

#-----------------------------------------------------------------------------------------#

def plot_density_model(density_map, output_dir='figure_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(density_map, cmap='gray', origin='upper')
    ax.set_title('Density Model')
    ax.axis('off')
    ax.set_xlabel('pixel-x')
    ax.set_ylabel('pixel-y')
    cbar = plt.colorbar(im, ax=ax, fraction=0.056, pad=0.04, orientation='horizontal')
    cbar.set_label('Density')
    plt.savefig(f'{output_dir}/density_model.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

def plot_gravitational_profile(x_obs_points, gravity_profile, output_dir='figure_out'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_obs_points, gravity_profile, 'b-', marker='o', markerfacecolor='green',
            markeredgewidth=2, markeredgecolor='black', markersize=10)
    ax.set_title('Gravitational Profile')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel(r'$\mathbf{g_{z}}$ (m/s²)')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, linestyle='--')
    plt.savefig(f'{output_dir}/gravitational_profile.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.show()

#-----------------------------------------------------------------------------------------#
