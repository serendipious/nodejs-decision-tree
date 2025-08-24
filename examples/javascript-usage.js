import DecisionTree from '../lib/decision-tree.js';

// Sample training data
const trainingData = [
  { color: "blue", shape: "square", size: "small", liked: false },
  { color: "red", shape: "square", size: "small", liked: false },
  { color: "blue", shape: "circle", size: "medium", liked: true },
  { color: "red", shape: "circle", size: "medium", liked: true },
  { color: "blue", shape: "hexagon", size: "large", liked: false },
  { color: "red", shape: "hexagon", size: "large", liked: false },
  { color: "yellow", shape: "hexagon", size: "small", liked: true },
  { color: "yellow", shape: "circle", size: "large", liked: true }
];

// Test data
const testData = [
  { color: "blue", shape: "hexagon", size: "medium", liked: false },
  { color: "yellow", shape: "circle", size: "small", liked: true }
];

// Features to use for classification
const features = ["color", "shape", "size"];
const target = "liked";

// Create and train the decision tree
const dt = new DecisionTree(target, features);
dt.train(trainingData);

// Make predictions
const sample1 = { color: "blue", shape: "hexagon", size: "medium" };
const sample2 = { color: "yellow", shape: "circle", size: "small" };

const prediction1 = dt.predict(sample1);
const prediction2 = dt.predict(sample2);

console.log(`Prediction for ${JSON.stringify(sample1)}: ${prediction1}`);
console.log(`Prediction for ${JSON.stringify(sample2)}: ${prediction2}`);

// Evaluate accuracy
const accuracy = dt.evaluate(testData);
console.log(`Accuracy on test data: ${(accuracy * 100).toFixed(1)}%`);

// Export the model
const modelJson = dt.toJSON();
console.log('Model exported successfully');

// Import the model to a new instance
const newDt = new DecisionTree(modelJson);
const prediction3 = newDt.predict(sample1);
console.log(`Prediction from imported model: ${prediction3}`);

// Access static properties
console.log('Node types:', DecisionTree.NODE_TYPES);

// Example with ES6 import syntax (if using bundlers)
// import DecisionTree from 'decision-tree';
// const dt = new DecisionTree(target, features);

// For CommonJS environments, you can use dynamic imports:
// const DecisionTree = await import('decision-tree');
// const dt = new DecisionTree.default(target, features);
