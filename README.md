# AsturHackers
After importing the data provided by the German Geosciences Research Center, we narrowed them down to those that temporally coincide with the magnetic field recorded by DSCOVR, which we also imported. As for the pre-processing of the data, we reduce it to 3h intervals and compose it into a single dataset. Then, we normalize the data, disregard the most irregular ones and apply a moving average to minimize noise. 

We realigned the inputs such that each Kp matches the magnetic field of previous intervals and split into training and validation data. We define the model, based on a sequential neural network with an input layer, a hidden layer of 32 neurons and an output layer, which returns a single value, the predicted Kp. We trained the model and evaluated its performance.
