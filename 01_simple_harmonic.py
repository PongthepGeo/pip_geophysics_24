#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs')
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
#-----------------------------------------------------------------------------------------#

# Define the frequency of the wave
frequency = 5
# Define the speed of the wave (in meters per second)
speed = 10
# Generate a time array from 0 to 1 second, with 500 points
time = np.linspace(0, 1, 500)
# Convert time to distance
distance = speed * time 
# Calculate the angle for the cosine and sine functions
x = 2 * np.pi * frequency * time
# Compute the cosine and sine values
cosine_values = np.cos(x)
sine_values = np.sin(x)

U.simple_harmonic_plot(time, cosine_values, sine_values)

#-----------------------------------------------------------------------------------------#