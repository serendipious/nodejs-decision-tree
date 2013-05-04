Decision Tree for NodeJS
========================

This module contains the NodeJS Implementation of Decision Tree using [ID3 Algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

### Installation
	npm install decision-tree

### Usage
	// Require Decision Tree Module
	var DecisionTree = require('decision-tree');
	
	// Datasets
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
	
	var test_data = [
		{"color":"blue", "shape":"hexagon", "liked":false},
		{"color":"red", "shape":"hexagon", "liked":false},
		{"color":"yellow", "shape":"hexagon", "liked":true},
		{"color":"yellow", "shape":"circle", "liked":true}
	];
	
	// Classifier setup
	var class_name = "liked";
	var features = ["color", "shape"];
	
	// Train model with Decision Tree
	var dt = new DecisionTree(training_data, class_name, features);
	
	// Predict class label for an instance
	var predicted_class = dt.predict({
		color: "blue",
		shape: "hexagon"
	});
	
	// Evaluate model on a dataset
	var accuracy = dt.evaluate(test_data);
