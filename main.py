import pandas as pd
import numpy as np

# Use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

train = pd.DataFrame(pd.read_csv('./titanic/train.csv'))
test = pd.DataFrame(pd.read_csv('./titanic/test.csv'))


class Decision_Node:
	def __init__(self, question, true_branch, false_branch):
		self.question = question
		self.true_branch = true_branch
		self.false_branch = false_branch


class Leaf:
	def __init__(self, rows):
		unique_labels = list(set(rows))
		self.predictions = {}
		for unique_label in unique_labels:
			self.predictions[unique_label] = rows.count(unique_label)




def replaceNANbyMean(column):
	val = column.mean(skipna=True)
	column = column.fillna(val)

	return column


def informationGain(left, right, current_uncertainty):

	p = float(len(left)) / (len(left) + len(right))

	return current_uncertainty - p * calculateGini(list(left['label'])) - (1 - p) * calculateGini(list(right['label']))



def makeQuestionAndParition(data, key, value):

	if str(value).isnumeric():
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
			true_Data, false_Data = data.loc[data[key] == value], data.loc[data[key] != value]
		else:
			true_Data, false_Data, nan = data.loc[data[key] == value], data.loc[data[key] != value], data.loc[
				data[key].isnull()]
			false_Data = pd.concat([false_Data, nan])

	return true_Data, false_Data, question


def findBestSplit(data):
	best_gain = 0
	best_question = None
	best_true = None
	best_false = None

	# calculates impurity
	current_uncertainity = calculateGini(list(data['label']))

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
				best_gain, best_question, best_true, best_false = gain, question, true_Data, false_Data


	return best_gain, best_question, true_Data, false_Data


def calculateGini(labels_list):
	impurity = 1
	unique_labels = list(set(labels_list))
	length = len(labels_list)

	for unique_label in unique_labels:
		count = labels_list.count(unique_label)
		prob = count / length
		impurity -= prob ** 2

	# 0.5 impurity means equally different labels. Same amount on 1s and 0s
	# 0 impurity means all are same type

	return impurity


def buildDecisionTree(data):
	# Find info gain and best question at this node.
	info_gain, question, true_Data, false_Data = findBestSplit(data)

	if info_gain == 0:
		return Leaf(list(data['label']))

	True_answers = buildDecisionTree(true_Data)
	False_answers = buildDecisionTree(false_Data)

	return Decision_Node(question, true_Data, false_Data)


if __name__ == "__main__":
	data = {'id': [0, 1, 2, 3, 4, 5], 'height': [10, 20, 40, 50, 100, 20], 'label': [0, 0, 0, 1, 1, 0]}
	df = pd.DataFrame(data)
	trainingData = df[['height', 'label']].copy()
	#print(buildDecisionTree(trainingData))

	Tree = buildDecisionTree(trainingData)

	def print_tree(node, spacing=""):
		"""World's most elegant tree printing function."""

		# Base case: we've reached a leaf
		if isinstance(node, Leaf):
			print(spacing + "Predict", node.predictions)
			return

		# Print the question at this node
		print(spacing + str(node.question))

		# Call this function recursively on the true branch
		print(spacing + '--> True:')
		print_tree(node.true_branch, spacing + "  ")

		# Call this function recursively on the false branch
		print(spacing + '--> False:')
		print_tree(node.false_branch, spacing + "  ")

	print_tree(Tree)



