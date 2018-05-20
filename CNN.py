from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialise the CNN
classifier = Sequential()

# 1. Add Convolution layer - this describes the feature maps, args = no. of features, rows, columns
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# 2. Add Pooling layer - this reduces the size of the feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 3. Add Flattening layer - take pooled feature map and flatten it to single dimension
classifier.add(Flatten())

# 4. Construct an ANN for training with the above layers
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])