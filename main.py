import mask_clothes
model = mask_clothes.Model(threshold=0.7)

ROOT_DIR = 'Output/'
for x in range(1, 4):
    img, masked_image, label_type, label, score, complete = model.run(IMG_DIR='Images/mask' + str(x) + '.jpg')
    if complete is True:
        for y in range(len(label)):
            directory = ROOT_DIR + label_type[y] + '/' + str(x) + '_' + label[y] + '.jpeg'
            masked_image[y].save(directory)
