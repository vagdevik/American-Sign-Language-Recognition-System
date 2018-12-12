## Project : American Sign Language Recognition System ##

** 4 Development Stages: **
1.  Got a dataset which contains 400 images of each class. Trained a VGG16 using transfer learning technique.
	Acheived 100% accuracy on this dataset. Problem is that all the images are structured without any
	real world background or practical components.
2.  Searched for a better dataset and got one which is from Kaggle. This contains 24 classes of
	alphabets(except J and Z) and special characters like space, delete, etc. Thus worked using only A,B,C.
	The VGG16 architechture above is very complex. Hence wanted to experiment using less complex model.
	Thus we trained an AlexNet(8 layers) from scratch on A,B,C classes. Achieved 80% accuracy.
3.  To improve the accuracy on this new dataset, trained VGG16 using transfer learning. Achieved 97% accuracy.
	Also did the real-time recognition through the input web-cam video.
4.  Increased the number of classes 5 and trained the VGG16 (using transfer learning) on 5 random images(B,C,L,V,W). 
	Achieved an Accuracy 99%.
Data link : https://www.kaggle.com/grassknoted/asl-alphabet
