# Decision Tree for Node.js

This Node.js module implements a Decision Tree using the [ID3 Algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

## Installation

**Requires Node.js 20 or higher** (ES modules support required)

```bash
npm install decision-tree
```

## TypeScript Support

This module is written in TypeScript and provides full type definitions. The compiled JavaScript maintains full backward compatibility with existing Node.js and browser projects that support ES modules.

**Note:** This package uses ES modules (`"type": "module"`), so CommonJS `require()` is not supported.

### TypeScript Usage

```typescript
import DecisionTree from 'decision-tree';
import RandomForest from 'decision-tree/random-forest';

// Full type safety for training data
interface TrainingData {
  color: string;
  shape: string;
  liked: boolean;
}

const training_data: TrainingData[] = [
  {"color":"blue", "shape":"square", "liked":false},
  {"color":"red", "shape":"square", "liked":false},
  {"color":"blue", "shape":"circle", "liked":true},
  {"color":"red", "shape":"circle", "liked":true}
];

// Decision Tree
const dt = new DecisionTree('liked', ['color', 'shape']);
dt.train(training_data);
const prediction = dt.predict({ color: "blue", shape: "hexagon" });

// Random Forest
const rf = new RandomForest('liked', ['color', 'shape'], {
  nEstimators: 100,
  maxFeatures: 'sqrt',
  randomState: 42
});
rf.train(training_data);
const rfPrediction = rf.predict({ color: "blue", shape: "hexagon" });
```

## Usage

### Import the module

```js
import DecisionTree from 'decision-tree';
```

**Important:** This package uses ES modules only. CommonJS `require()` is not supported.

### Prepare training dataset

```js
const training_data = [
  {"color":"blue", "shape":"square", "liked":false},
  {"color":"red", "shape":"square", "liked":false},
  {"color":"blue", "shape":"circle", "liked":true},
  {"color":"red", "shape":"circle", "liked":true},
  {"color":"blue", "shape":"hexagon", "liked":false},
  {"color":"red", "shape":"hexagon", "liked":false},
  {"color":"yellow", "shape":"hexagon", "liked":true},
  {"color":"yellow", "shape":"circle", "liked":true}
];
```

### Prepare test dataset

```js
const test_data = [
  {"color":"blue", "shape":"hexagon", "liked":false},
  {"color":"red", "shape":"hexagon", "liked":false},
  {"color":"yellow", "shape":"hexagon", "liked":true},
  {"color":"yellow", "shape":"circle", "liked":true}
];
```

### Setup Target Class used for prediction

```js
const class_name = "liked";
```

### Setup Features to be used by decision tree

```js
const features = ["color", "shape"];
```

### Create decision tree and train the model

**Method 1: Separate instantiation and training**
```js
const dt = new DecisionTree(class_name, features);
dt.train(training_data);
```

**Method 2: Instantiate and train in one step**
```js
const dt = new DecisionTree(training_data, class_name, features);
```

**Note:** Method 2 returns a new instance rather than modifying the current one. This is equivalent to:
```js
const dt = new DecisionTree(class_name, features);
dt.train(training_data);
```

### Predict class label for an instance

```js
const predicted_class = dt.predict({
  color: "blue",
  shape: "hexagon"
});
```

### Evaluate model on a dataset

```js
const accuracy = dt.evaluate(test_data);
```

### Export underlying model for visualization or inspection

```js
const treeJson = dt.toJSON();
```

**Note:** The exported model contains the tree structure but does not preserve the original training data. Only imported models have training data stored.

### Create a decision tree from a previously trained model

```js
const treeJson = dt.toJSON();
const preTrainedDecisionTree = new DecisionTree(treeJson);
```

### Import a previously trained model on an existing tree instance

```js
const treeJson = dt.toJSON();
dt.import(treeJson);
```

## Random Forest Usage

This package now includes a Random Forest implementation that provides better performance and reduced overfitting compared to single Decision Trees.

### Import Random Forest

```js
import RandomForest from 'decision-tree/random-forest';
```

### Basic Random Forest Usage

```js
const training_data = [
  {"color":"blue", "shape":"square", "liked":false},
  {"color":"red", "shape":"square", "liked":false},
  {"color":"blue", "shape":"circle", "liked":true},
  {"color":"red", "shape":"circle", "liked":true},
  {"color":"blue", "shape":"hexagon", "liked":false},
  {"color":"red", "shape":"hexagon", "liked":false},
  {"color":"yellow", "shape":"hexagon", "liked":true},
  {"color":"yellow", "shape":"circle", "liked":true}
];

const test_data = [
  {"color":"blue", "shape":"hexagon", "liked":false},
  {"color":"yellow", "shape":"circle", "liked":true}
];

const class_name = "liked";
const features = ["color", "shape"];

// Create and train Random Forest
const rf = new RandomForest(class_name, features);
rf.train(training_data);

// Make predictions
const predicted_class = rf.predict({
  color: "blue",
  shape: "hexagon"
});

// Evaluate accuracy
const accuracy = rf.evaluate(test_data);
console.log(`Accuracy: ${(accuracy * 100).toFixed(1)}%`);
```

### Random Forest Configuration

```js
const config = {
  nEstimators: 100,        // Number of trees (default: 100)
  maxFeatures: 'sqrt',     // Features per split: 'sqrt', 'log2', 'auto', or number
  bootstrap: true,         // Use bootstrap sampling (default: true)
  randomState: 42,         // Random seed for reproducibility
  maxDepth: undefined,     // Maximum tree depth
  minSamplesSplit: 2       // Minimum samples to split
};

const rf = new RandomForest(class_name, features, config);
rf.train(training_data);
```

### Random Forest Features

```js
// Get feature importance scores
const importance = rf.getFeatureImportance();
console.log('Feature importance:', importance);

// Get number of trees
const treeCount = rf.getTreeCount();
console.log(`Number of trees: ${treeCount}`);

// Get configuration
const config = rf.getConfig();
console.log('Configuration:', config);
```

### Random Forest Model Persistence

```js
// Export model
const modelJson = rf.toJSON();

// Import model
const newRf = new RandomForest(modelJson);

// Or import into existing instance
rf.import(modelJson);
```

### Random Forest vs Decision Tree

Random Forest typically provides:
- **Better accuracy** through ensemble learning
- **Reduced overfitting** via bootstrap sampling and feature randomization
- **More stable predictions** through majority voting
- **Feature importance** scores across the ensemble
- **Parallel training** capability for better performance

## XGBoost Usage

XGBoost (eXtreme Gradient Boosting) is a powerful gradient boosting algorithm that builds an ensemble of decision trees sequentially, where each tree corrects the errors of the previous ones.

### Basic XGBoost Usage

```js
import XGBoost from 'decision-tree/xgboost';

// Basic usage
const xgb = new XGBoost('liked', ['color', 'shape', 'size']);
xgb.train(training_data);

// Make predictions
const prediction = xgb.predict({ color: 'blue', shape: 'hexagon', size: 'medium' });

// Evaluate accuracy
const accuracy = xgb.evaluate(test_data);
console.log(`Accuracy: ${(accuracy * 100).toFixed(1)}%`);
```

### XGBoost Configuration

```js
const config = {
  nEstimators: 100,           // Number of boosting rounds (default: 100)
  learningRate: 0.1,          // Step size shrinkage (default: 0.1)
  maxDepth: 6,                // Maximum tree depth (default: 6)
  minChildWeight: 1,          // Minimum sum of instance weight in leaf (default: 1)
  subsample: 1.0,             // Fraction of samples for each tree (default: 1.0)
  colsampleByTree: 1.0,       // Fraction of features for each tree (default: 1.0)
  regAlpha: 0,                // L1 regularization (default: 0)
  regLambda: 1,               // L2 regularization (default: 1)
  objective: 'regression',    // Loss function: 'regression', 'binary', 'multiclass'
  earlyStoppingRounds: 10,    // Early stopping patience (default: undefined)
  randomState: 42,            // Random seed for reproducibility
  validationFraction: 0.2     // Fraction for validation set (default: 0.2)
};

const xgb = new XGBoost('liked', ['color', 'shape', 'size'], config);
xgb.train(training_data);
```

### XGBoost Features

```js
// Get feature importance scores
const importance = xgb.getFeatureImportance();
console.log('Feature importance:', importance);

// Get boosting history
const history = xgb.getBoostingHistory();
console.log('Training loss:', history.trainLoss);
console.log('Validation loss:', history.validationLoss);

// Get best iteration (useful with early stopping)
const bestIteration = xgb.getBestIteration();
console.log(`Best iteration: ${bestIteration}`);

// Get number of trees
const treeCount = xgb.getTreeCount();
console.log(`Number of trees: ${treeCount}`);

// Get configuration
const config = xgb.getConfig();
console.log('Configuration:', config);
```

### XGBoost Model Persistence

```js
// Export model
const modelJson = xgb.toJSON();

// Import model
const newXgb = new XGBoost(modelJson);

// Or import into existing instance
xgb.import(modelJson);
```

### Algorithm Comparison

| Feature | Decision Tree | Random Forest | XGBoost |
|---------|---------------|---------------|---------|
| **Algorithm** | Single tree (ID3) | Ensemble of trees | Gradient boosting |
| **Overfitting** | Prone to overfitting | Reduces overfitting | Best overfitting control |
| **Accuracy** | Good on simple data | Better on complex data | Best on complex data |
| **Interpretability** | Highly interpretable | Less interpretable | Least interpretable |
| **Training Time** | Fast | Medium | Slowest |
| **Prediction Time** | Fast | Medium | Fast |
| **Stability** | Less stable | More stable | Most stable |
| **Feature Selection** | All features | Random subset per tree | Random subset per tree |
| **Bootstrap Sampling** | No | Yes (by default) | Yes (configurable) |
| **Parallel Training** | No | Yes (trees independent) | No (sequential) |
| **Regularization** | No | No | Yes (L1, L2) |
| **Early Stopping** | No | No | Yes |
| **Learning Rate** | N/A | N/A | Yes |
| **Gradient Boosting** | No | No | Yes |

## Data Validation and Limitations

**Important:** This implementation is intentionally permissive and has limited validation:

- **Feature names:** Only validates that features is an array, not element types
- **Target column:** Does not validate that the target column exists in training data
- **Empty datasets:** Allows empty training datasets (may result in unexpected behavior)
- **Data types:** Accepts mixed data types without validation

For production use, ensure your data meets these requirements:
- Training data must be an array of objects
- Each object should contain the target column
- Feature values should be consistent across samples

## Error Handling

The package handles many edge cases gracefully but may fail silently in some scenarios:

```js
// This will work but may not produce expected results
const dt = new DecisionTree('nonexistent', ['feature1']);
dt.train([{ feature1: 'value1' }]); // Missing target column

// This will work but may not produce expected results  
const dt2 = new DecisionTree('target', ['feature1']);
dt2.train([]); // Empty dataset
```

## Test Coverage

This project maintains comprehensive test coverage to ensure reliability and correctness:

### Current Test Statistics
- **Total Tests:** 408 passing tests
- **Test Categories:** 15 comprehensive test suites covering Decision Trees, Random Forests, XGBoost, and Performance
- **Test Framework:** Mocha with TypeScript support
- **Coverage Areas:**
  - Core decision tree functionality
  - Random Forest ensemble learning
  - XGBoost gradient boosting
  - Data validation and sanitization
  - Edge cases and error handling
  - Performance and scalability
  - Type safety and interface validation
  - Model persistence and import/export
  - Prediction edge cases
  - ID3 algorithm correctness
  - Bootstrap sampling and feature selection
  - Majority voting and ensemble prediction

### Test Suites

| Test Suite | Description | Test Count |
|------------|-------------|------------|
| **Data Validation & Sanitization** | Input validation, feature validation, data type handling | 12 tests |
| **Decision Tree Basics** | Core functionality, initialization, training, prediction | 9 tests |
| **Edge Cases & Error Handling** | Empty datasets, missing features, invalid inputs | 8 tests |
| **Sample Dataset Tests** | Real-world dataset validation (Tic-tac-toe, Voting, Object Evaluation) | 7 tests |
| **ID3 Algorithm Tests** | Entropy calculations, feature selection, tree structure | 9 tests |
| **Model Persistence** | Import/export functionality, data integrity | 15 tests |
| **Performance & Scalability** | Large datasets, memory management, concurrent operations | 12 tests |
| **Prediction Edge Cases** | Missing features, unknown values, data type mismatches | 12 tests |
| **Type Safety & Interface Validation** | TypeScript type checking, interface consistency | 10 tests |
| **Reported Bugs** | Regression tests for previously reported issues | 2 tests |
| **Random Forest Basics** | Core Random Forest functionality, configuration, training | 10 tests |
| **Random Forest Configuration** | Different parameter combinations and edge cases | 9 tests |
| **Random Forest Bootstrap Sampling** | Bootstrap sampling with and without replacement | 3 tests |
| **Random Forest Feature Selection** | Random feature selection strategies | 4 tests |
| **Random Forest Ensemble Prediction** | Majority voting and prediction stability | 3 tests |
| **Random Forest Feature Importance** | Feature importance calculation and normalization | 3 tests |
| **Random Forest Model Persistence** | Export/import functionality for Random Forest models | 3 tests |
| **Random Forest Edge Cases** | Edge cases specific to Random Forest implementation | 15 tests |
| **Random Forest Performance** | Performance testing with large numbers of estimators | 2 tests |
| **Random Forest on Sample Datasets** | Real-world dataset validation with Random Forest | 3 tests |
| **Random Forest Utility Functions** | Bootstrap sampling, feature selection, majority voting utilities | 20 tests |
| **XGBoost Basics** | Core XGBoost functionality, configuration, training | 10 tests |
| **XGBoost Configuration** | Different parameter combinations and edge cases | 11 tests |
| **XGBoost Gradient Boosting** | Gradient boosting iterations and loss tracking | 3 tests |
| **XGBoost Early Stopping** | Early stopping functionality and validation | 3 tests |
| **XGBoost Feature Importance** | Feature importance calculation for XGBoost | 3 tests |
| **XGBoost Model Persistence** | Export/import functionality for XGBoost models | 4 tests |
| **XGBoost Edge Cases** | Edge cases specific to XGBoost implementation | 5 tests |
| **XGBoost Performance** | Performance testing with large numbers of estimators | 2 tests |
| **XGBoost on Sample Datasets** | Real-world dataset validation with XGBoost | 3 tests |
| **XGBoost Loss Functions** | Loss functions (MSE, Logistic, Cross-Entropy) | 15 tests |
| **XGBoost Gradient Boosting Utils** | Gradient boosting utility functions | 8 tests |
| **XGBoost Edge Cases - Empty Datasets** | Empty and invalid dataset handling | 7 tests |
| **XGBoost Edge Cases - Configuration** | Configuration edge cases and validation | 20 tests |
| **XGBoost Edge Cases - Prediction** | Prediction edge cases and validation | 9 tests |
| **XGBoost Edge Cases - Model Persistence** | Model persistence edge cases | 9 tests |
| **XGBoost Edge Cases - Feature Importance** | Feature importance edge cases | 3 tests |
| **XGBoost Edge Cases - Boosting History** | Boosting history edge cases | 3 tests |
| **XGBoost Edge Cases - Performance** | Performance edge cases | 4 tests |
| **Performance Tests - Decision Tree** | Decision Tree performance benchmarks | 6 tests |
| **Performance Tests - Random Forest** | Random Forest performance benchmarks | 7 tests |
| **Performance Tests - XGBoost** | XGBoost performance benchmarks | 8 tests |
| **Performance Tests - Algorithm Comparison** | Cross-algorithm performance comparison | 4 tests |
| **Performance Tests - Edge Cases** | Performance edge cases and stress tests | 3 tests |

### Performance Benchmarks

The library includes comprehensive performance tests to ensure all algorithms meet speed requirements:

- **Decision Tree**: < 100ms training, < 10ms prediction
- **Random Forest**: < 500ms training, < 50ms prediction  
- **XGBoost**: < 1000ms training, < 20ms prediction
- **Memory Usage**: < 50MB for large datasets
- **Scalability**: Linear scaling with dataset size and tree count

Performance tests cover:
- Training time benchmarks for small, medium, and large datasets
- Prediction speed tests with multiple iterations
- Memory usage monitoring for large datasets
- Algorithm comparison tests (Decision Tree vs Random Forest vs XGBoost)
- Concurrent operations and edge case performance
- Early stopping and regularization efficiency

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode (for development)
npm run test:watch

# Run performance tests specifically
npm test -- --grep "Performance Tests"

# Build and test
npm run build && npm test
```

### Test Quality Standards

- **100% Pass Rate:** All tests must pass before any code changes are merged
- **Comprehensive Coverage:** Tests cover happy paths, edge cases, and error scenarios
- **Performance Testing:** Includes tests for large datasets and memory efficiency
- **Type Safety:** Full TypeScript type checking and interface validation
- **Real-world Scenarios:** Tests with actual datasets (tic-tac-toe, voting records, etc.)

## Development

### Building from Source

This project is written in TypeScript. To build from source:

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Run tests
npm test

# Watch mode for development
npm run build:watch
```

## Windows Users

If you encounter issues with `npm test`, this project uses cross-env for cross-platform compatibility. The setup should work automatically, but if you encounter issues:

1. Ensure you're using Git Bash or WSL
2. Or use PowerShell/Command Prompt after running `npm install`

### Project Structure

- `src/` - TypeScript source files
- `lib/` - Compiled JavaScript output (generated)
- `tst/` - TypeScript test files
- `data/` - Sample datasets for testing

### Contributing

We welcome contributions to improve this decision tree implementation! To ensure high-quality contributions, please follow these guidelines:

#### Before You Start

1. **Check existing issues** - Look for open issues or discussions that might be related to your contribution
2. **Fork the repository** - Create your own fork to work on
3. **Create a feature branch** - Use a descriptive branch name like `feature/your-feature-name` or `fix/issue-description`

#### Development Workflow

1. **Make changes in the `src/` directory** - All source code changes should be in TypeScript
2. **Update tests in the `tst/` directory** - Add comprehensive tests for new functionality
3. **Run the build process** - Execute `npm run build` to compile TypeScript
4. **Run all tests** - Ensure `npm test` passes with 100% success rate
5. **Test your changes** - Verify your changes work as expected

#### Pull Request Requirements

To ensure high-quality contributions, all pull requests must include:

**Code Quality:**
- ✅ **TypeScript compliance** - All code must be properly typed and compile without errors
- ✅ **Test coverage** - New features must include comprehensive tests
- ✅ **Backward compatibility** - Changes should not break existing functionality
- ✅ **Performance consideration** - Large datasets and edge cases should be handled efficiently

**Documentation:**
- ✅ **Clear commit messages** - Use conventional commit format (e.g., `feat: add new feature`, `fix: resolve issue`)
- ✅ **Updated README** - If adding new features, update relevant documentation
- ✅ **Code comments** - Complex logic should be well-documented
- ✅ **Type definitions** - Ensure all public APIs have proper TypeScript definitions

**Testing Requirements:**
- ✅ **All tests pass** - The test suite must pass completely (currently 109 tests)
- ✅ **New test cases** - Add tests for new functionality in appropriate test files:
  - `decision-tree.ts` - Core functionality tests
  - `data-validation.ts` - Input validation and sanitization
  - `edge-cases.ts` - Edge cases and error handling
  - `performance-scalability.ts` - Performance and scalability tests
  - `type-safety.ts` - TypeScript type safety validation
- ✅ **Edge case coverage** - Test boundary conditions and error scenarios
- ✅ **Performance testing** - For performance-related changes, include benchmarks

**Code Style:**
- ✅ **Consistent formatting** - Follow existing code style and patterns
- ✅ **ES modules** - Maintain ES module compatibility (no CommonJS)
- ✅ **Node.js 20+ compatibility** - Ensure compatibility with the minimum Node.js version
- ✅ **Lodash usage** - Use existing lodash utilities where appropriate

#### Pull Request Template

When creating a pull request, please include:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Performance impact assessed (if applicable)

## Checklist
- [ ] Code follows existing style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or clearly documented if intentional)
```

#### Review Process

- All pull requests require review and approval
- Maintainers will check code quality, test coverage, and documentation
- Feedback will be provided for any required changes
- Once approved, changes will be merged to the main branch

#### Getting Help

If you need help or have questions:
- Open an issue for discussion before starting work on large changes
- Check existing issues and discussions
- Review the test files to understand expected behavior patterns

## Why Node.js 20+?

This package requires Node.js 20 or higher because:
- **ES Modules:** Uses native ES module support (`"type": "module"`)
- **Modern Features:** Leverages ES2022 features for better performance
- **Import Assertions:** Uses modern import syntax for better compatibility
- **Performance:** Takes advantage of Node.js 20+ optimizations
