# TypeScript Migration Guide

This document explains the TypeScript conversion of the decision-tree module and how to use it.

## What Changed

The module has been converted from JavaScript to TypeScript while maintaining **100% backward compatibility**. Existing JavaScript projects will continue to work without any changes.

## New Features

### 1. Full TypeScript Support
- Complete type definitions for all methods and properties
- Interface definitions for training data and model structures
- Compile-time type checking for better development experience

### 2. Enhanced Development Experience
- Source maps for debugging
- Declaration files (`.d.ts`) for IDE support
- Better IntelliSense and autocomplete

### 3. Modern Build System
- TypeScript compiler with ES5 output for maximum compatibility
- Watch mode for development
- Clean build process

## File Structure

```
├── src/                    # TypeScript source files
│   └── decision-tree.ts   # Main implementation
├── lib/                   # Compiled JavaScript (generated)
│   ├── decision-tree.js   # Main module
│   ├── decision-tree.d.ts # Type definitions
│   └── *.map             # Source maps
├── examples/              # Usage examples
│   ├── typescript-usage.ts
│   └── javascript-usage.js
└── tst/                   # Test files
```

## Usage Examples

### TypeScript Usage

```typescript
import DecisionTree from 'decision-tree';

interface TrainingData {
  color: string;
  shape: string;
  liked: boolean;
}

const trainingData: TrainingData[] = [
  { color: "blue", shape: "square", liked: false },
  { color: "red", shape: "circle", liked: true }
];

const dt = new DecisionTree('liked', ['color', 'shape']);
dt.train(trainingData);

const prediction = dt.predict({ color: "blue", shape: "hexagon" });
```

### JavaScript Usage (Backward Compatible)

```javascript
const DecisionTree = require('decision-tree');

const trainingData = [
  { color: "blue", shape: "square", liked: false },
  { color: "red", shape: "circle", liked: true }
];

const dt = new DecisionTree('liked', ['color', 'shape']);
dt.train(trainingData);

const prediction = dt.predict({ color: "blue", shape: "hexagon" });
```

### ES6 Module Usage

```javascript
import DecisionTree from 'decision-tree';

const dt = new DecisionTree('liked', ['color', 'shape']);
// ... rest of the code
```

## Development Commands

```bash
# Install dependencies
npm install

# Build the project
npm run build

# Watch mode for development
npm run build:watch

# Run tests
npm test

# Run examples
npm run example:js    # JavaScript example
npm run example:ts    # TypeScript example

# Clean build artifacts
npm run clean
```

## Type Definitions

The module provides comprehensive TypeScript interfaces:

```typescript
interface TreeNode {
  type: string;
  name: string;
  alias: string;
  val?: any;
  gain?: number;
  sampleSize?: number;
  vals?: TreeNode[];
  child?: TreeNode;
  prob?: number;
}

interface DecisionTreeData {
  model: TreeNode;
  data: any[];
  target: string;
  features: string[];
}
```

## Migration for Existing Projects

### No Changes Required
If you're using the module in JavaScript, **no changes are needed**. The compiled JavaScript maintains the exact same API.

### Adding TypeScript Support
To add TypeScript support to your project:

1. Install the module: `npm install decision-tree`
2. Import with types: `import DecisionTree from 'decision-tree'`
3. Define interfaces for your data structures
4. Enjoy full type safety!

## Browser Compatibility

The compiled JavaScript is ES5 compatible and works in:
- All modern browsers
- Node.js 14+
- Legacy environments
- Bundlers (Webpack, Rollup, etc.)

## Performance

- **Zero runtime overhead** - TypeScript types are removed during compilation
- **Same performance** as the original JavaScript version
- **Smaller bundle size** when using modern bundlers (ES6 modules)

## Contributing

When contributing to the project:

1. Make changes in the `src/` directory
2. Run `npm run build` to compile
3. Ensure tests pass with `npm test`
4. The compiled JavaScript in `lib/` is automatically generated

## Troubleshooting

### TypeScript Errors
- Ensure you're importing from the correct path
- Check that your data structures match the expected interfaces
- Use type assertions if needed: `data as TrainingData[]`

### Build Issues
- Run `npm run clean` to remove old build artifacts
- Ensure TypeScript is installed: `npm install typescript`
- Check `tsconfig.json` for configuration issues

### Runtime Errors
- The compiled JavaScript is identical to the original
- Check that you're using the correct import/require syntax
- Verify your data format matches the expected structure

## Support

For issues or questions:
1. Check the existing test files in `tst/`
2. Review the examples in `examples/`
3. Open an issue on GitHub
4. Check the main README.md for usage documentation
