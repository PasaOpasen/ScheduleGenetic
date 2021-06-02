

import os

import glob

import numpy as np

import imageio


from PIL import Image


files = glob.glob("./for_gif/*.png")

numbers = np.array([int(os.path.basename(f).split('_')[0]) for f in files])

files = [files[i] for i in np.argsort(numbers)]


image_list = []
for file_name in files:
    #image = Image.open(file_name)
    #image.thumbnail((400, 400))
    #image.save(file_name)
    image_list.append(imageio.imread(file_name))


imageio.mimwrite('animation.mp4', image_list)



# import matplotlib.pyplot as plt 
# import matplotlib.image as mgimg
# from matplotlib import animation

# fig = plt.figure()

# # initiate an empty  list of "plotted" images 
# myimages = []

# #loops through available png:s
# for file_name in files:

#     img = mgimg.imread(file_name)
#     imgplot = plt.imshow(img)

#     # append AxesImage object to the list
#     myimages.append([imgplot])

# ## create an instance of animation
# my_anim = animation.ArtistAnimation(fig, myimages, interval=100, blit=True, repeat_delay=1000)

# my_anim.save("animation.gif")





