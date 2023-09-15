import imageio.v3 as imageio
import scipy.ndimage as ndi
# Create a histogram, binned ateach possible value
import imageio
import numpy as np
import matplotlib.pyplot as plt
# Load the hand radiograph

def task_1 ():

    image=imageio.volread('archive(5)',format='dicom')

    print("NAME : ",image.meta['PatientName'])
    print("GENDER : ",image.meta['PatientSex'])
    print("BIRTHDATE : ",image.meta['PatientBirthDate'])
    print("ID : ",image.meta['PatientID'])
    print("Image Position: ", image.meta['ImagePositionPatient'])
    print("Image Orientation: ", image.meta['ImageOrientationPatient'])

    # plt.volshow(image,cmap='gray')

    # plt.show()



    a0,a1,a2 =image.shape

    print("slice number:-\n\t",
          "Axial=",a0,"Slices\n\t",
          "Coronal=",a1,"Slices\n\t",
          "Sagittal=",a2,"Slices\n\t"
          )



    b0,b1,b2=image.meta['sampling']

    print("Sampling:-\n\t",
          "Axial=",b0,"mm\n\t",
          "Coronal=",b1,"mm\n\t",
          "Sagittal=",b2,"mm\n\t"
          )



    x = a0 * b0
    x1 = a1 * b1
    x2 = a2 * b2

    print("Field of view:-\n\t",
          "Axial=",x,"mm\n\t",
          "Coronal=",x1,"mm\n\t",
          "Sagittal=",x2,"mm\n\t"
          )


    axial = b1 / b2
    Sagittal = b0 / b1
    Coronal = b0 / b2

    print("Pixel Aspect Ratio:-\n\t",
          "Axial=",axial,
          "Coronal=",Coronal,
          "Sagittal=",Sagittal
          )


    def Slicing ( axial_sl , Coronal_sl , Sagittal_sl ):

        fig, ax = plt.subplots(nrows=1 ,ncols=3, figsize=(12,12))

        ax[0].imshow(image[axial_sl,:,:], cmap='gray',aspect=axial)
        ax[0].axis('off')
        ax[0].set_title(' figure 1 ')

        ax[1].imshow(image[:, Coronal_sl, :], cmap='gray', aspect=Coronal)
        ax[1].axis('off')
        ax[1].set_title(' figure 2 ')

        ax[2].imshow(image[:, :, Sagittal_sl], cmap='gray', aspect=Sagittal)
        ax[2].axis('off')
        ax[2].set_title(' figure 3 ')

        plt.show()


    Slicing(10,30,90)


def task_2 () :

    def hist():
        im = imageio.imread("WhatsApp Image 2023-05-15 at 1.54.54 PM.jpeg")
        print('Data type:', im.dtype)
        print('Min. value:', im.min())
        print('Max value:', im.max())
        # Plot the grayscale image
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        plt.show()
        hist = ndi.histogram(im, min=0, max=255, bins=256)
        plt.plot(hist)
        plt.suptitle("Histogram Before applying morphological operation ")
        plt.show()

        mask_bone = im > 110
        mask_dilate = ndi.binary_dilation(mask_bone, iterations=5)
        mask_closed = ndi.binary_closing(mask_bone)
        # Plot masked images
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(mask_bone * 255)
        axes[1].imshow(mask_dilate * 255)
        axes[2].imshow(mask_closed * 255)
        plt.show()

        weights = [[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]]
        im_filt = ndi.convolve(im, weights)
        figure, axis = plt.subplots(1, 2)
        axis[0].imshow(im)
        axis[1].imshow(im_filt)
        plt.show()

        im_s1 = ndi.gaussian_filter(im, sigma=1)
        im_s3 = ndi.gaussian_filter(im, sigma=3)
        # Draw bone masks of each image
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(im)
        axes[1].imshow(im_s1)
        axes[2].imshow(im_s3)
        plt.show()

        weights = [[+1, 0, -1],
                   [+1, 0, -1],
                   [+1, 0, -1]]
        # Convolve "im" with filter weights
        edges = ndi.convolve(im, weights)
        # Draw the image in color
        plt.imshow(edges, vmin=0, vmax=150)
        plt.suptitle("feature detection  ")
        plt.show()

        sobel_ax0 = ndi.sobel(im, axis=0)
        sobel_ax1 = ndi.sobel(im, axis=1)
        # Calculate edge magnitude
        edges = np.sqrt(np.square(sobel_ax0) +
                        np.square(sobel_ax1))
        # Plot edge magnitude
        plt.imshow(edges, cmap='gray', vmax=75)
        plt.suptitle("Edge detection  ")
        plt.show()

        im = imageio.imread('chest-220.dcm')
        im_filt = ndi.median_filter(im, size=3)
        mask_start = np.where(im_filt > 60, 1, 0)
        mask = ndi.binary_closing(mask_start)
        labels, nlabels = ndi.label(mask)
        print('Num. Labels:', nlabels)
        overlay = np.where(labels > 0, labels,
                           np.nan)
        plt.imshow(overlay, cmap='rainbow')
        plt.axis('off')
        plt.show()

        # ssssssssssssssssss

        bboxes = ndi.find_objects(labels == 5)
        print('Number of objects:', len(bboxes))
        print('Indices for first box:', bboxes[0])
        im_lv = im[bboxes[0]]
        plt.imshow(im_lv)
        plt.show()

        d1, d2 = im.meta['sampling']
        dpixels = d1 * d2
        npixels = ndi.sum(1, labels, index=5)
        # Calculate volume of label
        area = npixels * dpixels
        print("area ")

        mask = np.where(labels == 5, 1, 0)
        # In terms of voxels
        d = ndi.distance_transform_edt(mask)
        d.max()

        # In terms of space
        lv = np.where(labels == 5, 1, 0)
        dists = ndi.distance_transform_edt(lv, sampling=im.meta['sampling'])
        # Report on distances
        print('Max distance (mm):',
              ndi.maximum(dists))
        print('Max location:',
              ndi.maximum_position(dists))

        coms = ndi.center_of_mass(im, labels, index=5)
        print('Label 1 center:', coms)

    hist()


#
# print("1- Get information from image metadata")
# print("2- Image analysis operations")
# print("3- Lungs disease classification Model ")
# print("Enter your choice : ")
# input=input()
# if input == 1:
#     print(task_1())
# elif input == 2:
#     print(task_2())


menu = input ("Choose the option you want:\n"
    "1- Get information from image metadata\n"
    "2- Image analysis operations\n"
    "3- Lungs disease classification Model\n")

if menu == str("1"):
    task_1()
elif menu == str("2"):
    task_2()
elif menu == str("3"):
    print ("Option 3")