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

## Test Coverage

This project maintains comprehensive test coverage to ensure reliability and correctness:

### Current Test Statistics
- **Total Tests:** 109 passing tests
- **Test Categories:** 8 comprehensive test suites covering all aspects of the decision tree implementation
- **Test Framework:** Mocha with TypeScript support
- **Coverage Areas:**
  - Core decision tree functionality
  - Data validation and sanitization
  - Edge cases and error handling
  - Performance and scalability
  - Type safety and interface validation
  - Model persistence and import/export
  - Prediction edge cases
  - ID3 algorithm correctness

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

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode (for development)
npm run test:watch

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
