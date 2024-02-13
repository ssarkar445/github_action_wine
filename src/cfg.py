import dispatcher
TRAINING_DATA = '../input/train_folds.csv'
MODEL = dispatcher.MODELS['bagging']
IDENTIFIER = 'Id'
TARGET = 'quality'
FEATURES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

