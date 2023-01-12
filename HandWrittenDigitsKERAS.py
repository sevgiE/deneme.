from tensorflow as tf
import numpy as np

(trainx,trainy)(testx,testy)=tf.keras.datasets.mnist.load_data()
trainx=tf.keras.utils.normalize(trainx,axis=1)
testx=tf.keras.utils.normalize(testx,axis=1)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.Layers.Dense(128,activation="tf.nn.relu"))
model.add(tf.keras.layers.Danse(10,activation=tf.nn.softmax))

model.compile(optinizer=tf.keras.optimizers.Adam(learninhg_rate=0.001),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(x=trainx,y=trainy,epochs=10)
model.save("./model/model.h5")
model.save_weights("./model/model_weights.h5")

testKayip,testHassasiyet=model.evaluate(x=testx,y=testy)

print("Test Hassasiyeti:",testHassasiyet)



