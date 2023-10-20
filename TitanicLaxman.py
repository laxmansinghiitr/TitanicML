import csv
import random
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation

training_data = []
testing_data = []

TrainFilePath = 'titanic/train.csv'
TestFilePath = 'titanic/test.csv'
TestResultPath = 'titanic/gender_submission.csv'

# path_select = 'titanic/train.csv'        # filedialog.askopenfile(title='Select File containing Training Data...')
# print(path_select)


# Size of x is 32 fields for training and testing
def create_training_data(path):
    training_data.clear()
    with open(path, 'r') as fileTrain:
        reader = csv.reader(fileTrain)  # reader = csv.reader(file, delimiter = '\t') if Tab delimiter
        header = next(reader)
        for row in reader:  # iteration of each row
            x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Re-initialize x to 0

            y = int(row[2])
            if y == 1:  # for Pclass 1,2,3: 2 bit information
                x[0] = 1
            elif y == 2:
                x[1] = 1
            else:
                x[2] = 1

            if row[4] == 'male':  # for Sex M/F: 1 bit information
                x[3] = 1
            else:
                x[4] = 1

            try:
                y = int(row[5])
                if y <= 5:  # for Age 0-5, 6-10, 11-18, 19-25, 26-35, 36-45, 46-60, 60+: 3 bit information
                    x[5] = 1
                elif y <= 10:
                    x[6] = 1
                elif y <= 18:
                    x[7] = 1
                elif y <= 25:
                    x[8] = 1
                elif y <= 35:
                    x[9] = 1
                elif y <= 45:
                    x[10] = 1
                elif y <= 60:
                    x[11] = 1
                else:
                    x[12] = 1
            except:
                x[8] = 1

            y = int(row[6])
            if y == 0:  # for #SibSP 1,2,3,4,5,6,7-8: 3 bit information
                x[13] = 1
            elif y == 1:
                x[14] = 1
            elif y == 2:
                x[15] = 1
            elif y == 3:
                x[16] = 1
            elif y == 4:
                x[17] = 1
            elif y == 5:
                x[18] = 1
            elif y == 6:
                x[19] = 1
            elif y == 7:
                x[20] = 1
            else:
                x[21] = 1

            y = int(row[7])
            if y == 0:  # for Parch 1,2,3,4,5,6: 3 bit information
                x[22] = 1
            elif y == 1:
                x[23] = 1
            elif y == 2:
                x[24] = 1
            elif y == 3:
                x[25] = 1
            elif y == 4:
                x[26] = 1
            elif y == 5:
                x[27] = 1
            else:
                x[28] = 1

            if row[11] == 'S':  # | row[11] == 's':  # for Embarked S/C/Q: 1 bit information
                x[29] = 1
            elif row[11] == 'C':  # | row[11] == 'c':
                x[30] = 1
            else:       # elif row[11] == 'Q':  # | row[11] == 'q':
                x[31] = 1

            # SummeryList.append(x)  # append at summery list
            training_data.append([x, int(row[1])])
            # print(x)
            # print(row)
        print('Training Data...')
        print(training_data)
    fileTrain.close()


def create_testing_data(path, path_result):   # difference in training and testing is, testing index is one less than training
    summery_list = []
    testing_data.clear()
    with open(path, 'r') as file:
        reader = csv.reader(file)  # reader = csv.reader(file, delimiter = '\t') if Tab delimiter
        header = next(reader)
        for row in reader:  # iteration of each row
            x = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]  # Re-initialize x to 0

            y = int(row[1])
            if y == 1:  # for Pclass 1,2,3: 2 bit information
                x[0] = 1
            elif y == 2:
                x[1] = 1
            else:
                x[2] = 1

            if row[3] == 'male':  # for Sex M/F: 1 bit information
                x[3] = 1
            else:
                x[4] = 1

            try:
                y = int(row[4])
                if y <= 5:  # for Age 0-5, 6-10, 11-18, 19-25, 26-35, 36-45, 46-60, 60+: 3 bit information
                    x[5] = 1
                elif y <= 10:
                    x[6] = 1
                elif y <= 18:
                    x[7] = 1
                elif y <= 25:
                    x[8] = 1
                elif y <= 35:
                    x[9] = 1
                elif y <= 45:
                    x[10] = 1
                elif y <= 60:
                    x[11] = 1
                else:
                    x[12] = 1
            except:
                x[8] = 1

            y = int(row[5])
            if y == 0:  # for #SibSP 1,2,3,4,5,6,7-8: 3 bit information
                x[13] = 1
            elif y == 1:
                x[14] = 1
            elif y == 2:
                x[15] = 1
            elif y == 3:
                x[16] = 1
            elif y == 4:
                x[17] = 1
            elif y == 5:
                x[18] = 1
            elif y == 6:
                x[19] = 1
            elif y == 7:
                x[20] = 1
            else:
                x[21] = 1

            y = int(row[6])
            if y == 0:  # for Parch 1,2,3,4,5,6: 3 bit information
                x[22] = 1
            elif y == 1:
                x[23] = 1
            elif y == 2:
                x[24] = 1
            elif y == 3:
                x[25] = 1
            elif y == 4:
                x[26] = 1
            elif y == 5:
                x[27] = 1
            else:
                x[28] = 1

            if row[10] == 'S':  # | row[10] == 's':  # for Embarked S/C/Q: 1 bit information
                x[29] = 1
            elif row[10] == 'C':  # | row[10] == 'c':
                x[30] = 1
            else:       # elif row[10] == 'Q':  # | row[11] == 'q':
                x[31] = 1

            summery_list.append(x)  # append at summery list
            # print(x)
            # print(row)
        print('Testing summery:')
        print(summery_list)
    file.close()

    with open(path_result, 'r') as file:
        reader = csv.reader(file)  # reader = csv.reader(file, delimiter = '\t') if Tab delimiter
        header = next(reader)
        i = 0
        for row in reader:
            testing_data.append([summery_list[i], int(row[1])])
            i = i+1
    file.close()

    print('Testing Data...')
    print(testing_data)


create_training_data(path=TrainFilePath)
create_testing_data(path=TestFilePath, path_result=TestResultPath)

with open('training_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(training_data)
f.close()

with open('testing_data.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(testing_data)
f.close()

# # random.shuffle(training_data)
X_train = []
Y_train = []
for data in training_data:
    X_train.append(data[0])
    Y_train.append(data[1])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[32]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training...')
model.fit(X_train, Y_train, epochs=4)
file_name = 'model_for_titanicSurvivalLaxman_32_128_128_128_1.model'
model.save(file_name)

# random.shuffle(training_data)     # No effect
# X = []
# Y = []
# for data in training_data:
#     X.append(data[0])
#     Y.append(data[1])
# model.fit(X, Y, epochs=1)

model.summary()

print('Testing...')
X_test = []
Y_test = []
for data in testing_data:
    X_test.append(data[0])
    Y_test.append(data[1])

predictions = model.predict(X_test)
tf.nn.softmax(predictions).numpy()
# print('Prediction[0]')
# print(predictions[0])
# print(predictions)

# max = max(predictions)
# min = min(predictions)
# print(max)
# print(min)
avg = sum(predictions) / len(predictions)
print('Average: '+str(avg))

result = []
for item in predictions:
    if item > avg:
        result.append(1)
    else:
        result.append(0)

print("Predicted Survival (with average value):")
print(result)
print("Actual Survival:")
print(Y_test)

count = 0
for (i, j) in zip(result, Y_test):
    if i == j:
        count = count + 1

accuracy = count / len(Y_test)

print('Correct prediction: '+str(count)+' / '+str(len(Y_test)))
print('Accuracy: '+str(accuracy))
print('Accuracy above '+str(int(accuracy * 100))+'%')

# Predicted Survival (0.5 cutoff)
result.clear()
for item in predictions:
    if item > 0.49:
        result.append(1)
    else:
        result.append(0)

print("Predicted Survival (0.5 cutoff):")
print(result)
print("Actual Survival:")
print(Y_test)

count = 0
for (i, j) in zip(result, Y_test):
    if i == j:
        count = count + 1

accuracy = count / len(Y_test)

print('Correct prediction: '+str(count)+' / '+str(len(Y_test)))
print('Accuracy: '+str(accuracy))
print('Accuracy above '+str(int(accuracy * 100))+'%')