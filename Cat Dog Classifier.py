from tensorflow.python.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


image_gen = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
        height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,fill_mode='nearest'
        ,horizontal_flip=True,rescale=1/255)


train_generator = image_gen.flow_from_directory('000__DATASETS/Deep Learning Datasets/CATS_DOGS/train',
                    target_size=(150,150),batch_size=16)

test_generator = image_gen.flow_from_directory('000__DATASETS/Deep Learning Datasets/CATS_DOGS/test',
                    target_size=(150, 150), batch_size=16)

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),strides=2,input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(4,4),strides=2,input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(4,4),strides=2,input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
model.fit_generator(train_generator,epochs=1,steps_per_epoch=150)

#%%

