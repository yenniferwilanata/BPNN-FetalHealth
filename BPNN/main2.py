import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

csv_name = "fetal_health.csv"

def load_dataset(filename):
    df = pd.read_csv(filename)
    correlations = df.drop(columns=["fetal_health"]).corrwith(df["fetal_health"])
    print(correlations)
    relevant_columns = correlations[abs(correlations) >= 0.4].index
    feature = df[relevant_columns]
    inp = feature.shape[1]
    target = df[['fetal_health']]
    return feature, target, inp

feature, target, inp = load_dataset(csv_name)

minMaxScaler = MinMaxScaler()
feature = minMaxScaler.fit_transform(feature)

ordinalEncoder = OrdinalEncoder()
feature = ordinalEncoder.fit_transform(feature)

oneHotEncoder = OneHotEncoder(sparse=False)
target = oneHotEncoder.fit_transform(target)

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

layers = {
    'input': inp,
    'hidden1': inp-1,
    'hidden2': inp-1,
    'hidden3': 3,
    'output': 3
}

weight = {
    'inputtohidden1':tf.Variable(tf.random_normal([layers['input'], layers['hidden1']])),
    'hidden1tohidden2':tf.Variable(tf.random_normal([layers['hidden1'], layers['hidden2']])),
    'hidden2tohidden3':tf.Variable(tf.random_normal([layers['hidden2'], layers['hidden3']])),
    'hidden3tooutput':tf.Variable(tf.random_normal([layers['hidden3'], layers['output']]))
}

bias = {
    'hidden1': tf.Variable(tf.random_normal([layers['hidden1']])),
    'hidden2': tf.Variable(tf.random_normal([layers['hidden2']])),
    'hidden3': tf.Variable(tf.random_normal([layers['hidden3']])),
    'output': tf.Variable(tf.random_normal([layers['output']])),
}

feature_placeholder = tf.placeholder(tf.float32, [None, layers['input']])
target_placeholder = tf.placeholder(tf.float32, [None, layers['output']])

def feed_forward():
    y1 = tf.matmul(feature_placeholder, weight['inputtohidden1']) + bias['hidden1']
    y1Active = tf.nn.sigmoid(y1)
    y2 = tf.matmul(y1Active, weight['hidden1tohidden2']) + bias['hidden2']
    y2Active = tf.nn.sigmoid(y2)
    y3 = tf.matmul(y2Active, weight['hidden2tohidden3']) + bias['hidden3']
    y3Active = tf.nn.sigmoid(y3)
    y4 = tf.matmul(y3Active, weight['hidden3tooutput']) + bias['output']
    y4Active = tf.nn.sigmoid(y4)
    return y4Active

lr = 0.1
epoch = 5000
output = feed_forward()

error = tf.reduce_mean((0.5 * target_placeholder - output) ** 2)

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(error)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1, epoch+1):
        train_dictionary = {
            feature_placeholder: x_train,
            target_placeholder: y_train
        }

        session.run(optimizer, feed_dict = train_dictionary)
        loss = session.run(error, feed_dict = train_dictionary)
        if (i % 50 == 0):
            print(f'Loss = {loss}')
        
    matches = tf.equal(tf.argmax(target_placeholder, axis = 1), tf.argmax(output, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    test_dictionary = {
        feature_placeholder: x_test,
        target_placeholder: y_test
    }

    print(f'Accuracy = {session.run(accuracy, feed_dict = test_dictionary)}')
