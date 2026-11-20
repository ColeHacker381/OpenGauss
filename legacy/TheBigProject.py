"""
===============================================================================
ENGR 13300 Fall 2023

Program Description
    

Assignment Information
    Assignment:     Team Project1 - PY4
    Author:         Cole, hacker21@purdue.edu
    Team ID:        LC3 - 10 

Contributor:    Sasmit Pandey, pandey96@purdue.edu
 [repeat for each]
    My contributor(s) helped me:
    [x] understand the assignment expectations without
        telling me how they will approach it.
    [x] understand different ways to think about a solution
        without helping me plan my solution.
    [x] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.
    
    Contributor:    Addison Hall, Hall781@purdue.edu
 [repeat for each]
    My contributor(s) helped me:
    [x ] understand the assignment expectations without
        telling me how they will approach it.
    [ x] understand different ways to think about a solution
        without helping me plan my solution.
    [x ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.
    
    Contributor:    Madeline Young, young736@purdue.edu
 [repeat for each]
    My contributor(s) helped me:
    [x ] understand the assignment expectations without
        telling me how they will approach it.
    [ x] understand different ways to think about a solution
        without helping me plan my solution.
    [x ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.
    
ACADEMIC INTEGRITY STATEMENT
I have not used source code obtained from any other unauthorized
source, either modified or unmodified. Neither have I provided
access to my code to another. The project I am submitting
is my own original work.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def image_to_sourcearray(source):#This function returns the source array.
    return plt.imread(source)
    
def image_to_targetarray(target):#This function returns the target array.
    return plt.imread(target)

def grayscalesource(sourcearray):#grayscale the source image into one array
    r, g, b = sourcearray[:,:,0], sourcearray[:,:,1], sourcearray[:,:,2]
    graysource= 0.2989 * r + 0.5870 * g + 0.1140 * b
    return graysource.astype(np.uint8)

def grayscaletarget(targetarray):#grayscale the target image into one array
    r, g, b = targetarray[:,:,0], targetarray[:,:,1], targetarray[:,:,2]
    graytarget = 0.2898*r + 0.5870*g + 0.1140*b
    return graytarget.astype(np.uint8)
            
def guassianfilter():#This function creates the guassian filter.
    filter = np.ones((5,5))
    filter[0] = [1,4,7,4,1]
    filter[1] = [4,16,26,16,4]
    filter[2] = [7,26,41,26,7]
    filter[3] = [4,16,26,16,4]
    filter[4] = [1,4,7,4,1]
    guas = 1/273 * filter
    return guas


def guassianfilterapp(image):#This function applies the guassian filter to the orginial image.
    filter= guassianfilter()
    filteredgraysource = np.zeros(image.shape)
    for I in range (image.shape[0]):
        for J in range (image.shape[1]):
            acc= 0 
            for K in range (filter.shape[0]):
                for L in range (filter.shape[1]):
                    if I+K < image.shape[0] and J+L < image.shape[1]:
                        acc+= filter[K][L]*image[I+K][J+L]
            filteredgraysource[I][J] = acc
    return filteredgraysource

def upsample(image):#This function upsamples the original image.

    #Make an empty, upscales array
    upsample = np.zeros((2* image.shape[0], 2* image.shape[1]))
    r, c = upsample.shape
    
    #sub in zeros for values in the original array
    for I in range(r):
        for J in range(c):
            if I%2 == 0 and J%2 == 0:
                upsample[I][J] = image[int(I/2)][int(J/2)]
                
    #pad array
    Padimage = np.pad(upsample,pad_width = 1, constant_values= 0)
    Padimage[1:r+1, -1] = upsample[:, -2]
    Padimage[-1, 1:c + 1] = upsample[-2, :]
    Padimage[-1, -1] = upsample[-2, -2]
    
    paddedR, paddedC  = Padimage.shape
    
    #fill in middle values with the average of the corner values
    
    for I in range(paddedR):
        for J in range(paddedC):
            if I%2 == 0 and J%2 == 0 and I and J:
                corners = ([Padimage[I-1, J-1], Padimage[I-1, J+1], Padimage[I+1, J-1], Padimage[I+1, J+1]])
                average = np.mean(corners)
                
                Padimage[I, J] = average
    
    #sub in values in the first row and column
    Padimage[0:paddedR, 0] = Padimage[:, 2]
    Padimage[0, 0:paddedC] = Padimage[2, :]
    Padimage[0, 0] = Padimage[2, 2]
    
    #fill in remaining 0 values with the touching values
    for I in range(r):
        for J in range(c):
            if Padimage[I,J] == 0 and I+1 != paddedR and J+1 != paddedC:
                DURL = ([Padimage[I, J-1], Padimage[I-1, J], Padimage[I+1, J], Padimage[I, J+1]])
                average2 = np.mean(DURL)
                
                Padimage[I, J] = average2
            
    #Take out padding and append array
    upsample = Padimage[1: -1, 1: -1]
    return upsample

def targetmask(image, size, x, y):#This function creates the mask for the image that the original image will be cropped onto.
    mask = np.ones(image.shape)
    mask[y-size//2:y+size//2, x-size//2:x+size//2] = 0

    return mask#return mask for image
    
def sourcemask(image, size, x, y):#This function creates the mask for the original image.
    mask = np.zeros(image.shape)#This line creates an array with 1s.
    mask[y-size//2:y+size//2, x-size//2:x+size//2] = 1
    return mask#return mask of orignal image

def create_laplacian_pyramid(image, pyramid_levels):
    laplacian_pyramid = []
    for level in range(pyramid_levels):
        # Apply filtering and subsampling
        filtered_image = guassianfilterapp(image)
        plt.imshow(filtered_image, cmap = "gray")
        plt.show()
        subsampled_image = filtered_image[::2, ::2]
        if level == pyramid_levels: 
            laplacian_pyramid.append(subsampled_image)  # Add the lowest resolution image
        else:
            # Upsample the subsampled image and apply filtering
            upsampled_image = upsample(subsampled_image)
            upsampledfiltimage = guassianfilterapp(upsampled_image)
            plt.imshow(upsampledfiltimage)
            plt.show()
            
            # Calculate the Laplacian and add it to the pyramid
            laplacian = image - upsampledfiltimage
            laplacian_pyramid.append(laplacian)
        image = subsampled_image  # Move to the next level
    laplacian_pyramid.append(subsampled_image)
    return laplacian_pyramid
    
def blend_images(targetlaplacianpyramid, sourcelaplacianpyramid, graytargetmask, graysourcemask):
    blendedpyramid = []
    
   # Determine the number of levels to iterate over
    firstlevel = min(len(targetlaplacianpyramid), len(sourcelaplacianpyramid))
    
    for level in range(firstlevel):
        if level == firstlevel - 1:
            target_mask = upsample(targetlaplacianpyramid[level])
            target_mask = guassianfilterapp(target_mask)
            source_mask = upsample(sourcelaplacianpyramid[level])
            source_mask = guassianfilterapp(source_mask)          
            blendedimage = target_mask.copy()
            blendedimage[graytargetmask == 0] = source_mask[graysourcemask == 1]
        else:
            print(level)
            targetlevel = targetlaplacianpyramid[level]
            sourcelevel = sourcelaplacianpyramid[level]
            
            # Combine the subsampled masks
            blendedimage = targetlevel.copy()
            
            blendedimage[graytargetmask == 0] = sourcelevel[graysourcemask == 1]
            
            graytargetmask = graytargetmask[::2, ::2].astype(np.uint8)
            graysourcemask = graysourcemask[::2, ::2].astype(np.uint8)
                
            blendedpyramid.append(blendedimage)
            plt.imshow(blendedimage)
            plt.show()

        
        
    return blendedpyramid, graytargetmask, graysourcemask

def reconstruct_image(lappyramid, smallest_target_image, smallest_source_image, source_mask, target_mask):
    
    reconstructed_image = lappyramid[len(lappyramid)-1]
    smallest_target_image = upsample(smallest_target_image)
    smallest_source_image = upsample(smallest_source_image)
    smallest_target_image = guassianfilterapp(smallest_target_image)
    smallest_source_image = guassianfilterapp(smallest_source_image)

    smallest_target_image = smallest_target_image * target_mask
    smallest_source_image = smallest_source_image * source_mask
    
    
    
    i = len(lappyramid)-2
    
    while i >= 0:
       print('hello') 
        
    '''
    
    level = len(lappyramid)-1
    reconstructedimage = lappyramid[0]
    for level in range(level-1, -1, -1):
        reconstructedimage = upsample(reconstructedimage)
        reconfiltimage = guassianfilterapp(reconstructedimage)
        reconfiltimage += level
        '''
    return reconstructed_image


#243 247 obama

#rock 247 229

def main():#This is the main function
    source =  input("Please enter source image JPEG file (with extension): ")
    target =  input("Please enter target image JPEG file (with extension): ")
    if len(target) > len(source) or len(target[0]) > len(source[0]):#This if statement checks to see if the image entered is a JPEG file or not.
        print("Error: Target image is larger than source image")#If the image entered is not a JPEG file, the program prints out an error statment.
        main()
    else:
            #These are the variables for the functions.
            sourcearray = image_to_sourcearray(source)#This creates the original image array.
            targetarray = image_to_targetarray(target)#This creates the traget image array.
            graysource = grayscalesource(sourcearray)#This creates the grayscale array for the orginal image.
            graytarget = grayscaletarget(targetarray)#This creates the grayscale array for the target image.
            #schrute/scott(1200, 672)
            #Obama/Rock(768,1104)
            
            fig, (axis1,axis2) = plt.subplots(1, 2)
            axis1.imshow(graytarget, cmap ="gray")#This line shows the target image in gray.
            axis2.imshow(graysource, cmap ="gray")#This line shows the source image in gray.
            plt.show()#This line shows both the images together.
            # Define the new dimensions and resize
            #for schrute input (730, 320)
            #for scott input (774, 364)
            #for Obama input(243,247)
            #for Rock input (247,229)
            xs= int(input("Please input central placement x value for source picture: "))#The user enters the x value for the source image, which gets saved in a variable named xs.
            ys= int(input("Please input central placement y value for source picture: "))#The user enters the y value for the source image, which gets saved in a variable named ys.
            xt= int(input("Please input central placement x value for target picture: "))#The user enters the x value for the target image, which gets saved in a variable named xt.
            yt= int(input("Please input central placement y value for target picture: "))#The user enters the y value for the target image, which gets saved in a variable named yt.
            size = int(input("Please enter the size (in pixels) to cut from the source image and paste on the target image: "))
            #scrute/scott = 275
            #Obama/Rock = 185
            if size < 0 or size > len(graytarget) or size > len(graytarget[0]):#This if statment checks to see if the image size entered is a postive number or not.
                print("Error: sizing is greater than the target image or is a negative number.")#If the image size entered is not a postive number, the porgram prints out an error statment.
                main()
           
            else:
                choosedef= int(input("Please input the number of pyramid levels to use in the smoothing process: "))
                if choosedef <= 0:
                    print("Error: Please choose a positive number")
                    main()
                print("Please wait while your image loads. Thank you for your patience!" )
                
                targetlappyramid = create_laplacian_pyramid(graytarget, choosedef)
                sourcelappyramid = create_laplacian_pyramid(graysource, choosedef)
                for x in targetlappyramid:
                    plt.imshow(x, cmap='gray')
                    plt.show()
                for x in sourcelappyramid:
                    plt.imshow(x, cmap='gray')
                    plt.show()
                graytargetmask = targetmask(graytarget, size, xt, yt)
                graysourcemask = sourcemask(graysource, size, xs, ys)
                blendedpyramid, smallest_target_mask, smallest_source_mask = blend_images(targetlappyramid, sourcelappyramid, graytargetmask, graysourcemask)
                for x in blendedpyramid:
                    plt.imshow(x, cmap='gray')
                    plt.show()
                # Reconstruct the final blended image
                finalimage = reconstruct_image(blendedpyramid, targetlappyramid[len(targetlappyramid)-1], sourcelappyramid[len(sourcelappyramid)-1], smallest_source_mask, smallest_target_mask)
                
                plt.imshow(finalimage, cmap = "gray")
                plt.show()
#This calls the main function.
if __name__ == '__main__':
    main()
    