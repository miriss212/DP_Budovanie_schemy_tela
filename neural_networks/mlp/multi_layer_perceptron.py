# Imports
import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


from data_processing.data_loader import *

class MLPWrapper:
    def __init__(self, vector_length, output_length, firstHidden, secondHidden=110, activationFunc='relu', outputFunc='sigmoid'):
        input_shape = (vector_length,)
        print(f'Feature shape: {input_shape}')

        self.Threshold = 0.4

        # Create the model
        self.model = Sequential()
        self.model.add(Dense(vector_length, input_shape=input_shape, activation=activationFunc))
        self.model.add(Dense(firstHidden, activation=activationFunc))
        self.model.add(Dense(secondHidden, activation=activationFunc))
        self.model.add(Dense(output_length, activation=outputFunc))

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])

    def train_model(self, X, Y, eps=150, batchSize=10):
        history = self.model.fit(X,Y, epochs=eps, batch_size=batchSize, validation_split=0.2, verbose=1)

        print("Training finished. Metrics of the training data:")

    def evaluate(self, X, Y):
        # Test the model after training
        loss, acc, mse = self.model.evaluate(X, Y, verbose=1)
        print(f'Test results - Loss(binary crossentrophy): {loss} - Accuracy: {acc*100}% - Mean Square Err: {mse}')
    def make_prediction(self, vector):
        prediction = self.model.predict(vector, batch_size=1)
        predictTouch = sum([0 if i < self.Threshold else 1 for i in prediction[0]])
        return predictTouch

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tensorflow.keras.models.load_model(filepath)

if __name__ == "__main__":
    # Configuration options
    feature_vector_length = 32
    output_touch_length = 64
    num_classes = 2
    train_new = False
    """
    Training MLP filter on ../my_data_act.data
    """

    # Convert target classes to categorical ones
    #Y_train = to_categorical(trainY, num_classes)
    #Y_test = to_categorical(testY, num_classes)
    f = open("../my_data.data")
    data = json.load(f)
    f.close()

    X = np.array([i["proprio"] for i in data])
    Y = np.array([i["touch"] for i in data])

    split = int(len(X) * 0.8)

    X_train = X[:split, :]
    X_test = X[split:, :]

    Y_train = Y[:split, :]
    Y_test = Y[split:, :]

    if train_new:
        #Config #
        batch_size = 20
        episodes = 1000

        #########

        model = MLPWrapper(feature_vector_length, output_touch_length, 70)
        model.compile_model()
        model.train_model(X_train, Y_train, episodes, batch_size)
        model.evaluate(X_test, Y_test)
        print("END")
        model.save_model("trained_keras_model")
        print("MLP model saved successfully.")
    else:
        #just load and evaluate;
        model = MLPWrapper(feature_vector_length, 70, 1)
        model.load_model("trained_keras_model")
        errorCount = 0

        with open("../whole_model/test_my_data.data") as f:
            testData = json.load(f)
        separator = int(len(testData) * 0.3)
        testData = testData[:separator]

        for iCubData, i in zip(testData, range(separator)):
            input = X_test[i]
            expectedOutput = sum(iCubData["touch"]) != 0
            prediction = model.make_prediction(np.array([iCubData["proprio"]]))

            output = 0 if prediction < model.Threshold else 1
            if output != expectedOutput:
                print("ERROR. Expected {0}, but was {1} (touch value predicted: {2})".format(expectedOutput, output, prediction))
                errorCount += 1

        print("DONE. Accuracy: {0} %".format(100 - ((errorCount / len(X_test))*100)))
        print("END")

#BEST: 80% s : 32, 70, int(70*1.5), 1; batchsize = 30, eps=150