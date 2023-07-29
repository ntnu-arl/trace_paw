import matplotlib.pyplot as plt

# Define the parameter sizes and corresponding inference speeds
parameter_sizes40x30 = [15083, 19539, 25235, 39843]
parameter_sizes80x60 = [24407, 77139, 82835]
inference_speeds40x30 = [700, 770, 910, 1200]
inference_speeds80x60 = [1550, 2750, 2870]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10,5))


plt.title('Inference speed vs Model size')
# Plot the data
ax.plot(parameter_sizes40x30, inference_speeds40x30, label="Model_40x30", marker="o", linestyle='dashed')
ax.plot(parameter_sizes80x60, inference_speeds80x60, label="Model_80x60", marker="o", linestyle='dashed')
ax.grid(True)
# Set the x and y axis labels
ax.set_xlabel('Parameter size')
ax.set_ylabel(u'Inference speed (\u03bcs)')
ax.legend()


# Show the plot
plt.savefig(f'figure/Training/inferenceVsParams.svg')
plt.show()
