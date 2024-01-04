"""
This code is designed to analyze and train the machine to extract spatial and temporal features of video files (preferably with .Avi extension).
In this code, a dataset folder with labeled video files  with different human activities (we worked on dataset UCF101 of Kaggle )
must be placed in the path of the code .
This code first loads all the files and divides them into two categories: train and test.
Then, first, it obtains the labels from the names of the files.
Then for each video file divides it into frames  and puts each group of frames into a separate folder (Group Of Pictures) and names the first frame of each GOPs as "Keyframe".
Then it extracts spatial features from Keyframe and extracts temporal features from the difference of every 2 Keyframes in a row.
After that, these features are taken to a convolutional neural network and trained in this model
In the next phase, it asks for an address that contains some query videos (which is folder query) 
and after performing the above steps, predicts the files label of activity it is related to.
"""
