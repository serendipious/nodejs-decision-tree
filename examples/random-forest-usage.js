import RandomForest from '../lib/random-forest.js';

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

console.log('=== Random Forest Example ===\n');

// Example 1: Basic Random Forest usage
console.log('1. Basic Random Forest:');
const rf1 = new RandomForest(target, features);
rf1.train(trainingData);

const sample1 = { color: "blue", shape: "hexagon", size: "medium" };
const prediction1 = rf1.predict(sample1);
console.log(`Prediction for ${JSON.stringify(sample1)}: ${prediction1}`);

const accuracy1 = rf1.evaluate(testData);
console.log(`Accuracy on test data: ${(accuracy1 * 100).toFixed(1)}%`);
console.log(`Number of trees: ${rf1.getTreeCount()}\n`);

// Example 2: Random Forest with custom configuration
console.log('2. Random Forest with custom configuration:');
const config = {
  nEstimators: 50,
  maxFeatures: 'sqrt',
  randomState: 42,
  bootstrap: true
};

const rf2 = new RandomForest(target, features, config);
rf2.train(trainingData);

const sample2 = { color: "yellow", shape: "circle", size: "small" };
const prediction2 = rf2.predict(sample2);
console.log(`Prediction for ${JSON.stringify(sample2)}: ${prediction2}`);

const accuracy2 = rf2.evaluate(testData);
console.log(`Accuracy on test data: ${(accuracy2 * 100).toFixed(1)}%`);
console.log(`Number of trees: ${rf2.getTreeCount()}`);
console.log(`Configuration:`, rf2.getConfig());

// Example 3: Feature importance
console.log('\n3. Feature Importance:');
const importance = rf2.getFeatureImportance();
console.log('Feature importance scores:');
Object.entries(importance).forEach(([feature, score]) => {
  console.log(`  ${feature}: ${score.toFixed(4)}`);
});

// Example 4: Model persistence
console.log('\n4. Model Persistence:');
const modelJson = rf2.toJSON();
console.log('Model exported successfully');

// Import the model to a new instance
const rf3 = new RandomForest(modelJson);
const prediction3 = rf3.predict(sample1);
console.log(`Prediction from imported model: ${prediction3}`);

// Example 5: Different feature selection strategies
console.log('\n5. Different feature selection strategies:');

const strategies = [
  { name: 'sqrt', maxFeatures: 'sqrt' },
  { name: 'log2', maxFeatures: 'log2' },
  { name: 'auto', maxFeatures: 'auto' },
  { name: '2 features', maxFeatures: 2 }
];

strategies.forEach(strategy => {
  const rf = new RandomForest(target, features, { 
    nEstimators: 10, 
    maxFeatures: strategy.maxFeatures,
    randomState: 42 
  });
  rf.train(trainingData);
  const accuracy = rf.evaluate(testData);
  console.log(`${strategy.name}: ${(accuracy * 100).toFixed(1)}% accuracy`);
});

// Example 6: Bootstrap sampling comparison
console.log('\n6. Bootstrap vs No Bootstrap:');

const withBootstrap = new RandomForest(target, features, { 
  nEstimators: 10, 
  bootstrap: true,
  randomState: 42 
});
withBootstrap.train(trainingData);
const accuracyWithBootstrap = withBootstrap.evaluate(testData);

const withoutBootstrap = new RandomForest(target, features, { 
  nEstimators: 10, 
  bootstrap: false,
  randomState: 42 
});
withoutBootstrap.train(trainingData);
const accuracyWithoutBootstrap = withoutBootstrap.evaluate(testData);

console.log(`With bootstrap: ${(accuracyWithBootstrap * 100).toFixed(1)}% accuracy`);
console.log(`Without bootstrap: ${(accuracyWithoutBootstrap * 100).toFixed(1)}% accuracy`);

console.log('\n=== Random Forest Example Complete ===');
