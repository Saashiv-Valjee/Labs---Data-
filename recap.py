import tensorflow as tf
print("-------------------CODE OUTPUT STARTS HERE-------------------")
print("tensorflow version: ", tf.__version__)

mnist = tf.keras.datasets.mnist

'''
It is a dataset of 60,000 small square 28×28 pixel grayscale 
images of handwritten single digits between 0 and 9.
'''

(x_train,y_train),(x_test,y_test) = mnist.load_data()

# converting numbers to floats
x_train, x_test = x_train/255.0,x_test/255.0

# building a model 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# using the models to make predictions on a fraction of the train data
predictions = model(x_train[:1]).numpy()
print(predictions)

# convert to probabilities 
tf.nn.softmax(predictions).numpy()

# Loss Function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

'''
The loss is zero if the model is sure of the correct class.
This untrained model gives probabilities 
close to random (1/10 for each class), so the initial loss
should be close to -tf.math.log(1/10) ~= 2.3.
'''

print(loss_fn(y_train[:1], predictions).numpy())

'''
Before you start training, configure and compile the model using Keras Model.compile. 
Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, 
and specify a metric to be evaluated for the model by setting the metrics parameter 
to accuracy.
'''

model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

# Use the Model.fit method to adjust your model parameters and minimize the loss: 
model.fit(x_train,y_train,epochs=5)

# The Model.evaluate method checks the models performance, 
# usually on a "Validation-set" or "Test-set".
model.evaluate(x_test,  y_test, verbose=2)

# If you want your model to return a probability, 
# you can wrap the trained model, and attach the softmax to it
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))

