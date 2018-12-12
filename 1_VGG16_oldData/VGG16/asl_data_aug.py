import os


from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        
        
data_folder = '../ASL_Data_sample/'
files = os.listdir(data_folder)

for file_name in files:
	print(file_name)
	path = data_folder+file_name
	img = load_img(path)  
	x = img_to_array(img)  # creating a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # converting to a Numpy array with shape (1, 3, 150, 150)
	
	i = 0
	for batch in datagen.flow(x,save_to_dir='../Augumented_Sample_Data', save_prefix=file_name[0], save_format='jpg'):
		i += 1
		if i > 36:
			break 
