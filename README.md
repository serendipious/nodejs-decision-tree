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

### License
(The MIT License)

Copyright (c) 2009-2012 TJ Holowaychuk <tj@vision-media.ca>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.