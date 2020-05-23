##############
# Example Code (Smart Coordinator)
# fab@kakao.com
##############

# Import Module
import mask_clothes

# Make model with img_size, threshold, gpu_count, images_per_gpu (There are defaults)
model = mask_clothes.Model(img_size=512, threshold=0.7, gpu_count=1, images_per_gpu=1)

#####################
# Return Value (run-method):
# img = Original image (Image)
# masked_image = Result Image (list of Image)
# label_type = Upper, Lower, Whole (list)
# label = Specific category name (list)
# complete = Whether model detects items well
#####################

# Set your Root directory
ROOT_DIR = 'Result/'

# Ex1. Using return-values, you can save complete sets
for x in range(1, 20):
    img, masked_image, label_type, label, score, complete = model.run(IMG_DIR='Images/mask' + str(x) + '.jpg')
    if complete is True:
        for y in range(len(label)):
            directory = ROOT_DIR + label_type[y] + '/' + str(x) + '_' + label[y] + '.jpeg'
            masked_image[y].save(directory)

# Ex2. Using show(), you can see masked items
img, masked_image, label_type, label, score, complete = model.run(IMG_DIR='Images/mask1.jpg')
for x in masked_image:
    x.show()
