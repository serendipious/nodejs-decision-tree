Decision Tree for Node.js
========================

This Node.js module implements a Decision Tree using the [ID3 Algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

# [Installation](id:installation)
	npm install decision-tree

# [Usage](id:usage)

## Import the module

	var DecisionTree = require('decision-tree');

## Prepare training dataset

	var training_data = [
		{"color":"blue", "shape":"square", "liked":false},
		{"color":"red", "shape":"square", "liked":false},
		{"color":"blue", "shape":"circle", "liked":true},
		{"color":"red", "shape":"circle", "liked":true},
		{"color":"blue", "shape":"hexagon", "liked":false},
		{"color":"red", "shape":"hexagon", "liked":false},
		{"color":"yellow", "shape":"hexagon", "liked":true},
		{"color":"yellow", "shape":"circle", "liked":true}
	];

## Prepare test dataset

	var test_data = [
		{"color":"blue", "shape":"hexagon", "liked":false},
		{"color":"red", "shape":"hexagon", "liked":false},
		{"color":"yellow", "shape":"hexagon", "liked":true},
		{"color":"yellow", "shape":"circle", "liked":true}
	];

## Setup Target Class used for prediction

	var class_name = "liked";

## Setup Features to be used by decision tree

	var features = ["color", "shape"];

## Create decision tree and train the model

	var dt = new DecisionTree(class_name, features);
	dt.train(training_data);

Alternately, you can also create and train the tree when instantiating the tree itself:

	var dt = new DecisionTree(training_data, class_name, features);

## Predict class label for an instance

	var predicted_class = dt.predict({
		color: "blue",
		shape: "hexagon"
	});

## Evaluate model on a dataset

	var accuracy = dt.evaluate(test_data);

## Export underlying model for visualization or inspection

	var treeJson = dt.toJSON();

## Create a decision tree from a previously trained model

	var treeJson = dt.toJSON();
	var preTrainedDecisionTree = new DecisionTree(treeJson);

Alternately, you can also import a previously trained model on an existing tree instance, assuming the features & class are the same:

	var treeJson = dt.toJSON();
	dt.import(treeJson);
