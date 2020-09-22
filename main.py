import pandas as pd
import numpy as np

# Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')


# print(train.keys())


def replaceNANbyMean(column):
	val = column.mean(skipna=True)
	column = column.fillna(val)

	return column


def informationGain(left, right, current_uncertainty):

	p = float(len(left)) / (len(left) + len(right))

	return current_uncertainty - p * calculateGini(left['labels']) - (1 - p) * calculateGini(right['labels'])



def makeQuestionAndParition(data, key, value):
	if value.isnumeric():
		question = f'{key} >= {value}?'
		if len(data.loc[data[key].isnull()]) != 0:
			true_Data, false_Data = data.loc[data[key] >= value], data.loc[data[key] < value]
		else:
			true_Data, false_Data, nan = data.loc[data[key] >= value], data.loc[data[key] < value], data.loc[
				data[key].isnull()]
			false_Data = pd.concat([false_Data, nan])
	elif value != value:
		true_Data, false_Data = [], []
		question = ''
	else:
		question = f'{key} == {value}?'
		if len(data.loc[data[key].isnull()]) != 0:
			question = f'{key} >= {value}?'
			true_Data, false_Data = data.loc[data[key] == name], data.loc[data[key] != name]
		else:
			true_Data, false_Data, nan = data.loc[data[key] == name], data.loc[data[key] != name], data.loc[
				data[key].isnull()]
			false_Data = pd.concat([false_Data, nan])

	return true_Data, false_Data, question


def findBestSplit(data):
	best_gain = 0
	best_question = None

	# calculates impurity
	current_uncertainity = calculateGini(data['labels'])

	# key is the column name, column is the data in that column
	for key, column in data.items():

		# get all unique values in the column
		unique_values = list(set(row for row in column))

		for unique_value in unique_values:

			# generate a question according to value and column data
			true_Data, false_Data, question = makeQuestionAndParition(data, key, unique_value)

			if len(true_Data) == 0 or len(false_Data) == 0:
				continue

			gain = informationGain(true_Data, false_Data, current_uncertainity)

			if gain > best_gain:
				best_gain, best_question = gain, question

	return best_gain, best_question


def calculateGini(labels_list):
	impurity = 1
	unique_labels = list(set(labels_list))
	length = len(labels_list)

	for unique_label in unique_labels:
		count = labels_list.count(unique_label)
		prob = count / length
		impurity -= prob ** 2

	# 0.5 impurity means equally different labels.
	# 0 impurity means all are same type

	return impurity


def buildDecisionTree(data):
	# Find info gain and best question at this node.
	info_gain, question = findBestSplit(data)

	if info_gain == 0:
		return 'Leaf'

	# if info_gain == 0:

	# Split data to true answer and false answer

	# save best split and question

	# True answers = buildDecisionTree(answers)
	# False answers = buildDecisionTree(answers)
	pass


if __name__ == "__main__":
	data = {'id': [0, 1, 2, 3, 4, 5], 'height': [10, 20, 40, 50, 100, 20], 'label': [0, 0, 0, 1, 1, 0]}
	c = train['Name']
	name = c[0]
	true_Data, false_Data, nan = train.loc[c == name], train.loc[c != name], train.loc[c.isnull()]

	print(f'{len(train)}, {len(true_Data) + len(false_Data)}')
