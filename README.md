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

const dt = new DecisionTree('liked', ['color', 'shape']);
dt.train(training_data);

// Type-safe prediction
const prediction = dt.predict({ color: "blue", shape: "hexagon" });
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

### Project Structure

- `src/` - TypeScript source files
- `lib/` - Compiled JavaScript output (generated)
- `tst/` - TypeScript test files
- `data/` - Sample datasets for testing

### Contributing

When contributing, please:
1. Make changes in the `src/` directory (TypeScript source)
2. Update tests in the `tst/` directory (TypeScript tests)
3. Run `npm run build` to compile
4. Ensure all tests pass with `npm test`
5. The compiled JavaScript in `lib/` will be automatically generated

## Why Node.js 20+?

This package requires Node.js 20 or higher because:
- **ES Modules:** Uses native ES module support (`"type": "module"`)
- **Modern Features:** Leverages ES2022 features for better performance
- **Import Assertions:** Uses modern import syntax for better compatibility
- **Performance:** Takes advantage of Node.js 20+ optimizations
