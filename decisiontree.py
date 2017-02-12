# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 23:57:52 2017

@author: Harsha.Varun
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import _tree

def decision_tree_classifier(train_x, train_y):
	param_grid = {'max_depth': range(3, 10)}
	new_tree = DecisionTreeClassifier(max_depth = 5)
	new_tree.fit(train_x, train_y)
	return new_tree




def tree_to_code(tree, feature_names):
	tree_ = tree.tree_
	feature_name = [
		feature_names[i] for i in tree_.feature
		]
	rule_string = ["def tree({}):".format(", ".join(feature_names))]

	def recurse(node, depth):
		indent = "  " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			rule_string.append("{}if {} <= {}:".format(indent, name, threshold))
			recurse(tree_.children_left[node], depth + 1)
			rule_string.append("{}else:  # if {} > {}".format(indent, name, threshold))
			recurse(tree_.children_right[node], depth + 1)
		else:
			rule_string.append("{}return {}".format(indent, tree_.value[node]))
		return rule_string
	return recurse(0,1)

def compare_trees(tree1, tree2,feature_names):
	tree1_str = tree_to_code(tree1,feature_names)
	tree2_str = tree_to_code(tree2,feature_names)
	for i in range(len(tree1_str)):
		if tree1_str[i] != tree2_str[i]:
			return tree2_str[i]
	return 0