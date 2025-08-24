Decision Tree for Node.js
========================

This Node.js module implements a Decision Tree using the [ID3 Algorithm](http://en.wikipedia.org/wiki/ID3_algorithm)

# [Installation](id:installation)

**Requires Node.js 20 or higher**

    npm install decision-tree

## TypeScript Support

This module is written in TypeScript and provides full type definitions. The compiled JavaScript maintains full backward compatibility with existing Node.js and browser projects. **Requires Node.js 20+ for development and testing.**

### TypeScript Usage

```typescript
import DecisionTree from 'decision-tree';

// Or with CommonJS
const DecisionTree = require('decision-tree');

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

# [Usage](id:usage)

## Import the module

```js
var DecisionTree = require('decision-tree');
```

## Prepare training dataset

```js
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
```

## Prepare test dataset

```js
var test_data = [
  {"color":"blue", "shape":"hexagon", "liked":false},
  {"color":"red", "shape":"hexagon", "liked":false},
  {"color":"yellow", "shape":"hexagon", "liked":true},
  {"color":"yellow", "shape":"circle", "liked":true}
];
```

## Setup Target Class used for prediction

```js
var class_name = "liked";
```

## Setup Features to be used by decision tree

```js
var features = ["color", "shape"];
```

## Create decision tree and train the model

```js
var dt = new DecisionTree(class_name, features);
dt.train(training_data);
```

Alternately, you can also create and train the tree when instantiating the tree itself:

```js
var dt = new DecisionTree(training_data, class_name, features);
```

## Predict class label for an instance

```js
var predicted_class = dt.predict({
  color: "blue",
  shape: "hexagon"
});
```

## Evaluate model on a dataset

```js
var accuracy = dt.evaluate(test_data);
```

## Export underlying model for visualization or inspection

```js
var treeJson = dt.toJSON();
```

## Create a decision tree from a previously trained model

```js
var treeJson = dt.toJSON();
var preTrainedDecisionTree = new DecisionTree(treeJson);
```

Alternately, you can also import a previously trained model on an existing tree instance, assuming the features & class are the same:

```js
var treeJson = dt.toJSON();
dt.import(treeJson);
```

# [Development](id:development)

## Building from Source

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

## Project Structure

- `src/` - TypeScript source files
- `lib/` - Compiled JavaScript output (generated)
- `tst/` - Test files
- `data/` - Sample datasets for testing

## Contributing

When contributing, please:
1. Make changes in the `src/` directory
2. Run `npm run build` to compile
3. Ensure all tests pass with `npm test`
4. The compiled JavaScript in `lib/` will be automatically generated
