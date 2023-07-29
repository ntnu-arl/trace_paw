import matplotlib.pyplot as plt

# Define the parameter sizes and corresponding inference speeds
parameter_sizes40x30 = [15083, 19539 ,25235, 39843]
parameter_sizes80x60 = [24407, 77139 ,82835]
accuracy40x30 = [1.421, 1.427 ,1.387, 1.38]
accuracy80x60 = [1.373, 1.307,1.285]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10,5))

plt.title('MAE vs Model size')
# Plot the data
ax.plot(parameter_sizes40x30, accuracy40x30, label="Model_40x30", marker="o", linestyle='dashed')
ax.plot(parameter_sizes80x60, accuracy80x60, label="Model_80x60", marker="o", linestyle='dashed')
ax.grid(True)
# Set the x and y axis labels
ax.set_xlabel('Parameter size')
ax.set_ylabel(u'MAE (N)')
ax.legend()


# Show the plot
plt.savefig(f'figure/Training/accuracyVsParams.svg')
plt.show()
