Decision Tree for NodeJS
========================

This module contains the NodeJS Implementation of Decision Tree using [ID3 Algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

# Table Of Contents
* [`Installation`](#installation)
* [`Usage`](#usage)
	* [`Import Module`](#installation)
	* [`Prepare`](#installation)
	* [`Installation`](#installation)

# [Installation](id:installation)
	npm install decision-tree

# [Usage](id:usage)

* Import the module:

		var DecisionTree = require('decision-tree');
	
* Prepare training dataset:

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
	
* Prepare test dataset:

		var test_data = [
			{"color":"blue", "shape":"hexagon", "liked":false},
			{"color":"red", "shape":"hexagon", "liked":false},
			{"color":"yellow", "shape":"hexagon", "liked":true},
			{"color":"yellow", "shape":"circle", "liked":true}
		];
	
* Setup Target Class used for prediction:
	
		var class_name = "liked";

* Setup Features to be used by decision tree:

		var features = ["color", "shape"];
	
* Create decision tree and train model:
	
		var dt = new DecisionTree(training_data, class_name, features);
	
* Predict class label for an instance:

		var predicted_class = dt.predict({
			color: "blue",
			shape: "hexagon"
		});
	
* Evaluate model on a dataset:
	
		var accuracy = dt.evaluate(test_data);

* Export underlying model for visualization or inspection:

		var treeModel = dt.toJSON();