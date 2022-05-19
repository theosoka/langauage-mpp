import os
import random
import re
import numpy as np
from string import ascii_lowercase


class Perceptron:

    def __init__(self, lang):
        self.W = np.random.randn(27)
        self.X = np.asarray(input_vectors[::])
        self.D = []
        self.lang = lang
        self.rate = 0.1

    def predict(self, x):
        output = self.W[0]
        for i in range(len(x) - 1):
            output += self.W[i + 1] * x[i]
        return 1. if output >= 0. else 0.

    def desired_outputs(self, desired_answers):
        for lang in desired_answers:
            values = []
            for value in lang:
                values.append(1 if value == self.lang else 0)
            self.D.append(values)
        self.D = np.asarray(self.D)

    def train(self):
        for n in range(200):
            for i in range(len(self.X)):
                for j in range(len(self.X[i])):
                    prediction = self.predict(self.X[i][j])
                    error = self.D[i][j] - prediction
                    self.W[0] = self.W[0] + self.rate * error
                    for k in range(len(self.W) - 1):
                        self.W[k + 1] = self.W[k + 1] + self.rate * error * self.X[i][j][k]


def get_files(directory):
    files = []
    for l in os.scandir(directory):
        if l.is_dir():
            texts = []
            for filename in os.scandir(l.path):
                if filename.is_file():
                    texts.append(filename.path)
            files.append(texts)
    return files


def relative_frequency_of_letters(text):
    frequency = []
    num_of_chars = len(text)
    for x in ascii_lowercase:
        num_of_the_letter = text.count(x)
        frequency.append(num_of_the_letter / num_of_chars)
    frequency.append(1.)
    return frequency


def one_language_inputs(lang):
    inputs = []
    for file in lang:
        fin = open(file, 'rt', encoding='utf8')
        content = fin.read()
        fin.close()
        content = modify_text(content)
        inputs.append(relative_frequency_of_letters(content))
    return inputs


def modify_text(text_):
    return re.sub("[^a-zA-Z]+", "", text_).lower()


input_vectors = []
training_answers = []
test_answers = []
test_vectors = []
languages = []
perceptrons = []
predicted_answers = []

training_files = get_files('training')
test_files = get_files('test')

for language in training_files:
    num_files = len(language)
    languages.append(re.search(r'\\([a-z]+)\\', language[0]).group(1))
    training_answers.append([re.search(r'\\([a-z]+)\\', language[0]).group(1) for i in range(0, num_files)])

for language in test_files:
    num_files = len(language)
    test_answers.append([re.search(r'\\([a-z]+)\\', language[0]).group(1) for i in range(0, num_files)])
test_answers = np.asarray(test_answers)

for training_language in training_files:
    input_vectors.append(one_language_inputs(training_language))
input_vectors = np.asarray(input_vectors)

for test_lang in test_files:
    test_vectors.append(one_language_inputs(test_lang))
test_vectors = np.asarray(test_vectors)

for language in languages:
    perceptrons.append(Perceptron(language))

for perceptron in perceptrons:
    perceptron.desired_outputs(training_answers)
    perceptron.train()


for i in range(len(test_vectors)):
    for j in range(len(test_vectors[i])):
        temp = []
        for perceptron in perceptrons:
            temp.append(perceptron.predict(test_vectors[i][j]))
        if temp.count(1.) == 1:
            predicted_answers.append(languages[temp.index(1.)])
        else:
            predicted_answers.append(languages[random.randint(0, len(languages) - 1)])

test_answers = test_answers.reshape(test_answers.size)
c = 0
for i in range(len(predicted_answers)):
    if predicted_answers[i] == test_answers[i]:
        c += 1
accuracy = c / len(predicted_answers)
print(f'accuracy = {accuracy:.2f}')


def precision(lang):
    corr_all = 0
    for i in range(len(predicted_answers)):
        if predicted_answers[i] == lang and test_answers[i] == lang:
            corr_all += 1
    predicted = predicted_answers.count(lang)
    return corr_all / predicted


def recall(lang):
    corr_all = 0
    for i in range(len(predicted_answers)):
        if predicted_answers[i] == lang and test_answers[i] == lang:
            corr_all += 1
    predicted = np.count_nonzero(test_answers == lang)
    return corr_all / predicted


for language in languages:
    p = precision(language)
    r = recall(language)
    f_measure = 1 / (1 / p + 1 / r)
    print(f'{language} perceptron:\n'
          f'   *precision = {p:.2f}\n'
          f'   *recall = {r:.2f}\n'
          f'   *F-measure = {f_measure:.2f}')

while True:
    values = input()
    if values == 'stop':
        break
    else:
        text = modify_text(values)
        freq = relative_frequency_of_letters(text)
        temp = []
        for perceptron in perceptrons:
            temp.append(perceptron.predict(freq))
        if temp.count(1.) == 1:
            print(languages[temp.index(1.)])
        else:
            print(languages[random.randint(0, len(languages) - 1)])
