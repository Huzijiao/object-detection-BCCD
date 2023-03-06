
# importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import os


# 根据csv文件在图片上标注三色框
def filterFiles(directoryPath, extension):
    """
        This function filters the format files with the selected extension in the directory

        Args:
            directoryPath (str): relative path of the directory that contains text files
            extension (str): extension file

        Returns:
            The list of filtered files with the selected extension
    """
    relevant_path = directoryPath
    included_extensions = [extension]
    file_names = [file1 for file1 in os.listdir(relevant_path) if
                  any(file1.endswith(ext) for ext in included_extensions)]
    numberOfFiles = len(file_names)
    listParams = [file_names, numberOfFiles]
    return listParams


[image_names, numberOfFiles] = filterFiles("data/JPEGImages", "jpg")

trainRCNN = pd.read_csv('data/nms_for_plot.csv', sep=",", header=None)
trainRCNN.columns = ['filename','xmin', 'xmax', 'ymin', 'ymax','cell_type']
trainRCNN.head()

for imageFileName in image_names:
    fig = plt.figure()
    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])  # adding X and Y axes from 0 to 1 for each direction
    plt.axis('off')

    # read and plot the image
    image = plt.imread('data/JPEGImages/' + imageFileName)
    plt.imshow(image)
    i = 0
    # iterating over the image for different objects
    for _, row in trainRCNN[trainRCNN.filename == imageFileName].iterrows():
        xmin = float(row.xmin)
        xmax = float(row.xmax)
        ymin = float(row.ymin)
        ymax = float(row.ymax)
        width = xmax - xmin
        height = ymax - ymin
        ClassName = row.cell_type
        # assign different color to different classes of objects
        if row.cell_type == '2':
            ax.annotate('RBC', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
        elif row.cell_type == '3':
            ax.annotate('WBC', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='b', facecolor='none')
        elif row.cell_type == '1':
            ax.annotate('Platelets', xy=(xmax - 40, ymin + 20))
            rect = patches.Rectangle((xmin, ymin), width, height, edgecolor='g', facecolor='none')
        else:
            print("nothing")
        ax.add_patch(rect)
        if not os.path.exists("imagesBox"):
            os.makedirs("imagesBox")
        fig.savefig('imagesBox/' + imageFileName, dpi=90, bbox_inches='tight')
        print("已画出一个框")
    plt.close()
    print("ImageName: " + imageFileName + " is saved in imagesBox folder")
    if imageFileName == "BloodImage_00002.jpg":
        break
print("PLOTBOX COMPLETED!")
