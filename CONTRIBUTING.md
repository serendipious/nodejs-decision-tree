# Contributing to Decision Tree Library

Thank you for your interest in contributing to this machine learning library! This guide will help you contribute high-quality code that maintains the project's standards.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Project Structure](#project-structure)
- [Common Issues](#common-issues)

## Getting Started

### Prerequisites

- Node.js 20+ or Bun 1.0+
- npm, yarn, or bun package manager
- Git
- Basic understanding of TypeScript
- Familiarity with machine learning concepts

### Why Use Bun?

Bun is fully supported and offers several advantages for development:

- **Faster Installation**: Package installation is significantly faster than npm
- **Built-in TypeScript**: No need for ts-node or additional TypeScript tooling
- **Faster Test Execution**: Bun's test runner is optimized for speed
- **Better Performance**: Generally faster execution for JavaScript/TypeScript code
- **Compatible**: Works with all existing npm packages and scripts

### Setting Up Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/nodejs-decision-tree.git
   cd nodejs-decision-tree
   ```

3. **Install dependencies**:
   ```bash
   # Using npm
   npm install
   
   # Using Bun (recommended for faster installation)
   bun install
   ```

4. **Build the project**:
   ```bash
   # Using npm
   npm run build
   
   # Using Bun
   bun run build
   ```

5. **Run tests** to ensure everything works:
   ```bash
   # Using npm
   npm test
   
   # Using Bun (faster test execution)
   bun test
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Use descriptive branch names:
- `feature/add-new-algorithm`
- `fix/performance-issue`
- `docs/update-readme`
- `test/add-edge-cases`

### 2. Make Your Changes

- **Source Code**: Make changes in the `src/` directory
- **Tests**: Add/update tests in the `tst/` directory
- **Documentation**: Update README.md and other docs as needed

### 3. Build and Test

#### Using npm
```bash
# Build the project
npm run build

# Run all tests
npm test

# Run specific test categories
npm test -- --grep "Decision Tree"
npm test -- --grep "Performance Tests"

# Run tests in watch mode (for development)
npm run test:watch
```

#### Using Bun (Recommended)
```bash
# Build the project
bun run build

# Run all tests (Bun has built-in TypeScript support)
bun test

# Run specific test categories
bun test --grep "Decision Tree"
bun test --grep "Performance Tests"

# Run tests in watch mode (for development)
bun test --watch

# Note: Bun can run TypeScript files directly without compilation
# For development, you can also run: bun test tst/*.ts
```

### 4. Commit Your Changes

Use conventional commit messages:

```bash
git add .
git commit -m "feat: add new feature description"
git commit -m "fix: resolve issue description"
git commit -m "test: add tests for new functionality"
git commit -m "docs: update documentation"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Quality Standards

### TypeScript Requirements

- **Full Type Safety**: All code must be properly typed
- **No `any` Types**: Avoid using `any` unless absolutely necessary
- **Interface Definitions**: Define clear interfaces for all data structures
- **Generic Types**: Use generics where appropriate for reusability

```typescript
// ✅ Good
interface TrainingData {
  features: string[];
  target: string;
  data: Record<string, any>[];
}

// ❌ Bad
const data: any = [...];
```

### Error Handling

- **Graceful Degradation**: Handle edge cases gracefully
- **Clear Error Messages**: Provide meaningful error messages
- **Input Validation**: Validate inputs at appropriate boundaries

```typescript
// ✅ Good
if (!Array.isArray(data)) {
  throw new Error('Training data must be an array');
}

// ❌ Bad
// Silent failure or unclear error
```

### Performance Considerations

- **Efficient Algorithms**: Use efficient algorithms and data structures
- **Memory Management**: Avoid memory leaks and excessive memory usage
- **Scalability**: Consider performance with large datasets
- **Benchmarking**: Include performance tests for new features

## Testing Requirements

### Test Coverage

- **100% Pass Rate**: All 421 tests must pass before merging
- **New Feature Tests**: Add comprehensive tests for new functionality
- **Edge Case Coverage**: Test boundary conditions and error scenarios
- **Performance Tests**: Include performance benchmarks where applicable (basic performance tests only)

### Test Categories

Add tests to appropriate files:

| Test File | Purpose | When to Use |
|-----------|---------|-------------|
| `decision-tree.ts` | Core Decision Tree functionality | New Decision Tree features |
| `random-forest.ts` | Random Forest functionality | New Random Forest features |
| `xgboost.ts` | XGBoost functionality | New XGBoost features |
| `performance-tests.ts` | Basic performance benchmarks | Performance-related changes |
| `edge-cases.ts` | Edge cases and error handling | Error handling improvements |
| `data-validation.ts` | Input validation | Input validation changes |
| `type-safety.ts` | TypeScript type safety | Type system changes |

### Test Structure

```typescript
describe('Feature Name', function() {
  it('should handle normal case', () => {
    // Arrange
    const input = createTestData();
    
    // Act
    const result = functionUnderTest(input);
    
    // Assert
    assert.strictEqual(result, expectedValue);
  });

  it('should handle edge case', () => {
    // Test edge cases
  });

  it('should throw error for invalid input', () => {
    // Test error conditions
  });
});
```

### Performance Testing

For performance-related changes, add basic benchmarks (avoid extensive performance tests):

```typescript
it('should perform within reasonable time limits', () => {
  const startTime = Date.now();
  // Perform operation
  const endTime = Date.now();
  
  const executionTime = endTime - startTime;
  assert.ok(executionTime < MAX_ALLOWED_TIME, 
    `Operation took ${executionTime}ms, expected < ${MAX_ALLOWED_TIME}ms`);
});
```

**Note**: Extensive performance and memory usage tests have been removed to focus on core functionality. Only add basic performance tests when necessary.

## Pull Request Process

### Before Submitting

- [ ] All 421 tests pass (`npm test` or `bun test`)
- [ ] Code builds without errors (`npm run build` or `bun run build`)
- [ ] New features have comprehensive tests
- [ ] Documentation is updated
- [ ] Code follows style guidelines
- [ ] No breaking changes (or clearly documented)

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Test addition/improvement

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed
- [ ] Performance impact assessed (if applicable - basic tests only)
- [ ] Edge cases tested

## Checklist
- [ ] Code follows existing style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or clearly documented if intentional)
- [ ] TypeScript types are properly defined
- [ ] Error handling is appropriate
- [ ] Performance considerations addressed
```

### Review Process

1. **Automated Checks**: CI will run tests and build checks
2. **Code Review**: Maintainers will review code quality and functionality
3. **Feedback**: Address any feedback or requested changes
4. **Approval**: Once approved, changes will be merged

## Code Style Guidelines

### General Guidelines

- **Consistent Formatting**: Follow existing code style
- **Clear Naming**: Use descriptive variable and function names
- **Comments**: Add comments for complex logic
- **ES Modules**: Use ES module syntax (no CommonJS)

### TypeScript Specific

```typescript
// ✅ Good - Clear interfaces
interface AlgorithmConfig {
  nEstimators: number;
  maxDepth?: number;
  randomState?: number;
}

// ✅ Good - Proper typing
function trainModel(data: TrainingData[], config: AlgorithmConfig): void {
  // Implementation
}

// ❌ Bad - Unclear types
function train(data: any, options: any): any {
  // Implementation
}
```

### File Organization

- **One class per file**: Keep classes in separate files
- **Shared utilities**: Put shared code in `src/shared/`
- **Clear exports**: Use named exports where appropriate
- **Import organization**: Group imports logically

### Error Messages

```typescript
// ✅ Good - Clear and actionable
throw new Error('Training data must be an array of objects with at least one sample');

// ❌ Bad - Vague
throw new Error('Invalid data');
```

## Project Structure

```
src/
├── decision-tree.ts          # Main Decision Tree class
├── random-forest.ts          # Random Forest implementation
├── xgboost.ts               # XGBoost implementation
└── shared/                  # Shared utilities
    ├── types.ts             # TypeScript type definitions
    ├── loss-functions.ts    # Loss function implementations
    └── gradient-boosting.ts # Gradient boosting utilities

tst/                         # Test files
├── decision-tree.ts         # Decision Tree tests
├── random-forest.ts         # Random Forest tests
├── xgboost.ts              # XGBoost tests
├── performance-tests.ts    # Performance benchmarks
└── ...                     # Other test files

examples/                    # Usage examples
├── javascript-usage.js     # JavaScript examples
├── typescript-usage.ts     # TypeScript examples
└── ...                     # Algorithm-specific examples
```

## Common Issues

### Build Errors

**Issue**: TypeScript compilation errors
**Solution**: 
```bash
# Using npm
npm run build
# Fix any TypeScript errors shown

# Using Bun
bun run build
# Fix any TypeScript errors shown
```

### Test Failures

**Issue**: Tests failing
**Solution**:
```bash
# Using npm
npm test
# Fix failing tests or update expectations

# Using Bun
bun run test
# Fix failing tests or update expectations
```

### Import/Export Issues

**Issue**: Module import/export errors
**Solution**: Ensure you're using ES module syntax:
```typescript
// ✅ Correct
import DecisionTree from 'decision-tree';

// ❌ Incorrect (CommonJS)
const DecisionTree = require('decision-tree');
```

### Performance Issues

**Issue**: New code is too slow
**Solution**: 
- Add performance tests
- Optimize algorithms
- Consider memory usage
- Use appropriate data structures

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Ask for help during code review
- **Documentation**: Check existing documentation and examples

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors list

Thank you for contributing to this project! Your efforts help make this library better for everyone.
