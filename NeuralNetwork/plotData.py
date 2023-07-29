import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math




def plot_tensorboard_data(filepath, files, plotname):
    # read the CSV file into a pandas dataframe
    df = pd.read_csv(filepath + files[0])
    df1 = pd.read_csv(filepath + files[1])
    df2 = pd.read_csv(filepath + files[2])
    # df3 = pd.read_csv(filepath + files[3])

    print(df['Value'].min())
    print(df1['Value'].min())
    print(df2['Value'].min())
    # print(df3['Value'].min())
    # plot the data
    plt.figure(figsize=(10,5))
    # plt.title('Validation Loss for Baseline models')
    plt.ylabel('MAE of normalized forces')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.plot(df['Step'],df['Value'])
    plt.plot(df1['Step'],df1['Value'])
    plt.plot(df2['Step'],df2['Value'])
    # plt.plot(df3['Step'],df3['Value'])

    plt.grid(True, ls="-")
    plt.legend([f'{files[0][:-4]}', f'{files[1][:-4]}', f'{files[2][:-4]}'])
    # plt.legend([f'{files[0][:-4]}', f'{files[1][:-4]}', f'{files[2][:-4]}', f'{files[3][:-4]}'])
    plt.savefig(f'figure/Training/{plotname}.svg')
    # df.plot()
    plt.show()



filepath = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/NeuralNetwork/data2/outputs/tensorlogs_csv/'
file1 = 'model_40x30_0.csv'
file2 = 'model_40x30_1.csv'
file3 = 'model_40x30_2.csv'
file4 = 'model_40x30_3.csv'
files = [file1, file2, file3, file4]

plotname = 'model_40x30'


filepath = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/NeuralNetwork/data2/outputs/tensorlogs_csv/'
file1 = 'model_80x60_0.csv'
file2 = 'model_80x60_1.csv'
file3 = 'model_80x60_2.csv'
file4 = 'model_80x60_3.csv'
files = [file1, file2, file3, file4]

plotname = 'model_80x60'

filepath = 'E:/Bruker/Dokumenter/Skole/Master/smart_paws/NeuralNetwork/data2/outputs/tensorlogs_csv/'
file1 = 'model_160x120_0.csv'
file2 = 'model_160x120_1.csv'
file3 = 'model_160x120_2.csv'
files = [file1, file2, file3]

plotname = 'model_160x120'

plot_tensorboard_data(filepath, files, plotname)

# # load data 
# dataPath = 'data2/outputs/'

# modelname = 'Model_80x60_3'

# df = pd.read_csv(dataPath + 'val_Model_60x80_1610241')

# # print(df)


# f_mag = (df["y_Fx"]**2 + df["y_Fy"]**2 + df["y_Fz"]**2)**(1/2)
# f_hat_mag = (df["y_hat_Fx"]**2 + df["y_hat_Fy"]**2 + df["y_hat_Fz"]**2)**(1/2)


# tot_error = (df["Error_Fx"] + df["Error_Fy"] + df["Error_Fz"])/3
# print((abs(f_mag-f_hat_mag)).mean())



# fig, ax = plt.subplots(figsize=(10,5))

# plt.title(f'{modelname}')
# ax.scatter(f_mag, abs(f_mag-f_hat_mag),  label = "Fx")
# ax.grid(True)
# ax.set_xlabel('Force vector magnitude')
# ax.set_ylabel('Absolute error [N]')

# plt.savefig(f'figure/Training/AE_{modelname}.svg')
# # ax.axis('equal')
# # ax.axis('square')
# # ax.scatter(f_hat_mag, label = "Fx_hat")

# # # Plot the data
# # ax.plot(parameter_sizes40x30, inference_speeds40x30, label="Model_40x30", marker="o", linestyle='dashed')
# # ax.plot(parameter_sizes80x60, inference_speeds80x60, label="Model_80x60", marker="o", linestyle='dashed')
# # ax.grid(True)
# # # Set the x and y axis labels
# # ax.set_xlabel('Parameter size')
# # ax.set_ylabel(u'Inference speed (\u03bcs)')
# # ax.legend()


# # ax[0,0].plot(df["Fx"], label='Fx')
# # ax[0,0].legend()

# # ax[0,1].plot(df2["Fy"], label='Fy')
# # ax[0,1].legend()
# # ax[0,2].plot(df2["Fz"], label='Fz')
# # ax[0,2].legend()
# # ax[1,0].plot(df2["roll"], label='Roll')
# # ax[1,0].legend()
# # ax[1,1].plot(df2["pitch"], label='Pitch')
# # ax[1,1].legend()
# # ax[1,2].plot(df2["yaw"], label='Yaw')
# # ax[1,2].legend()

# plt.show()