import assert from 'node:assert';
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = resolve(__filename, '..');

describe('Package Configuration Tests', function() {
  let packageJson: any;

  before(function() {
    const packagePath = resolve(__dirname, '..', 'package.json');
    packageJson = JSON.parse(readFileSync(packagePath, 'utf8'));
  });

  describe('Basic Package Information', function() {
    it('should have correct package name', function() {
      assert.strictEqual(packageJson.name, 'decision-tree');
    });

    it('should have valid version format', function() {
      assert.ok(/^\d+\.\d+\.\d+$/.test(packageJson.version), 
        `Version ${packageJson.version} should follow semver format`);
    });

    it('should have author information', function() {
      assert.ok(packageJson.author, 'Author should be specified');
      assert.strictEqual(packageJson.author, 'Ankit Kuwadekar');
    });

    it('should have correct repository configuration', function() {
      assert.ok(packageJson.repository, 'Repository should be specified');
      assert.strictEqual(packageJson.repository.type, 'git');
      assert.ok(packageJson.repository.url.includes('github.com'), 
        'Repository URL should point to GitHub');
    });
  });

  describe('Module Configuration', function() {
    it('should be configured as ES module', function() {
      assert.strictEqual(packageJson.type, 'module', 
        'Package should be configured as ES module');
    });

    it('should have correct main entry point', function() {
      assert.strictEqual(packageJson.main, './lib/decision-tree.js');
      assert.strictEqual(packageJson.types, './lib/decision-tree.d.ts');
      assert.strictEqual(packageJson.module, './lib/decision-tree.js');
    });

    it('should include correct files in package', function() {
      assert.ok(Array.isArray(packageJson.files), 'Files should be an array');
      assert.ok(packageJson.files.includes('lib'), 'Should include lib directory');
      assert.ok(packageJson.files.includes('src'), 'Should include src directory');
    });

    it('should require Node.js 20+', function() {
      assert.ok(packageJson.engines, 'Engines should be specified');
      assert.strictEqual(packageJson.engines.node, '>= 20.x');
    });
  });

  describe('Export Configuration', function() {
    it('should have exports object', function() {
      assert.ok(packageJson.exports, 'Exports should be configured');
      assert.strictEqual(typeof packageJson.exports, 'object');
    });

    it('should export main decision tree module', function() {
      const mainExport = packageJson.exports['.'];
      assert.ok(mainExport, 'Main export should exist');
      assert.strictEqual(mainExport.import, './lib/decision-tree.js');
      assert.strictEqual(mainExport.require, './lib/decision-tree.js');
      assert.strictEqual(mainExport.types, './lib/decision-tree.d.ts');
    });

    it('should export xgboost module', function() {
      const xgboostExport = packageJson.exports['./xgboost'];
      assert.ok(xgboostExport, 'XGBoost export should exist');
      assert.strictEqual(xgboostExport.import, './lib/xgboost.js');
      assert.strictEqual(xgboostExport.require, './lib/xgboost.js');
      assert.strictEqual(xgboostExport.types, './lib/xgboost.d.ts');
    });

    it('should export random-forest module', function() {
      const rfExport = packageJson.exports['./random-forest'];
      assert.ok(rfExport, 'Random Forest export should exist');
      assert.strictEqual(rfExport.import, './lib/random-forest.js');
      assert.strictEqual(rfExport.require, './lib/random-forest.js');
      assert.strictEqual(rfExport.types, './lib/random-forest.d.ts');
    });

    it('should export package.json', function() {
      assert.strictEqual(packageJson.exports['./package.json'], './package.json');
    });

    it('should have consistent export paths', function() {
      const exports = packageJson.exports;
      const expectedExports = ['.', './xgboost', './random-forest', './package.json'];
      
      expectedExports.forEach(exportPath => {
        assert.ok(exports[exportPath], `Missing export for ${exportPath}`);
      });

      // Check that all module exports point to lib directory
      Object.keys(exports).forEach(key => {
        if (key !== './package.json') {
          const exportConfig = exports[key];
          assert.ok(exportConfig.import.startsWith('./lib/'), 
            `${key} import should point to lib directory`);
          assert.ok(exportConfig.require.startsWith('./lib/'), 
            `${key} require should point to lib directory`);
          assert.ok(exportConfig.types.startsWith('./lib/'), 
            `${key} types should point to lib directory`);
        }
      });
    });
  });

  describe('Dependencies', function() {
    it('should have required dependencies', function() {
      assert.ok(packageJson.dependencies, 'Dependencies should exist');
      assert.ok(packageJson.dependencies.lodash, 'Lodash should be a dependency');
    });

    it('should have development dependencies', function() {
      assert.ok(packageJson.devDependencies, 'Dev dependencies should exist');
      
      const requiredDevDeps = [
        '@types/lodash',
        '@types/mocha', 
        '@types/node',
        'cross-env',
        'esm',
        'mocha',
        'ts-node',
        'typescript'
      ];

      requiredDevDeps.forEach(dep => {
        assert.ok(packageJson.devDependencies[dep], `Missing dev dependency: ${dep}`);
      });
    });
  });

  describe('Scripts Configuration', function() {
    it('should have required scripts', function() {
      assert.ok(packageJson.scripts, 'Scripts should exist');
      
      const requiredScripts = [
        'build',
        'test',
        'clean',
        'prepublishOnly'
      ];

      requiredScripts.forEach(script => {
        assert.ok(packageJson.scripts[script], `Missing script: ${script}`);
      });
    });

    it('should have correct build script', function() {
      assert.strictEqual(packageJson.scripts.build, 'tsc');
    });

    it('should have example scripts', function() {
      const exampleScripts = [
        'example:js',
        'example:ts', 
        'example:rf-js',
        'example:rf-ts',
        'example:xgb-js',
        'example:xgb-ts'
      ];

      exampleScripts.forEach(script => {
        assert.ok(packageJson.scripts[script], `Missing example script: ${script}`);
      });
    });
  });

  describe('Keywords and License', function() {
    it('should have relevant keywords', function() {
      assert.ok(Array.isArray(packageJson.keywords), 'Keywords should be an array');
      assert.ok(packageJson.keywords.length > 0, 'Should have keywords');
      
      const expectedKeywords = [
        'decision tree',
        'random forest', 
        'xgboost',
        'machine learning',
        'typescript'
      ];

      expectedKeywords.forEach(keyword => {
        assert.ok(packageJson.keywords.includes(keyword), 
          `Missing keyword: ${keyword}`);
      });
    });

    it('should have MIT license', function() {
      assert.strictEqual(packageJson.license, 'MIT');
    });
  });

  describe('File System Validation', function() {
    it('should have all exported files exist', function() {
      Object.keys(packageJson.exports).forEach(exportPath => {
        if (exportPath !== './package.json') {
          const exportConfig = packageJson.exports[exportPath];
          
          // Check if files actually exist
          const importPath = resolve(__dirname, '..', exportConfig.import);
          const typesPath = resolve(__dirname, '..', exportConfig.types);
          
          assert.ok(existsSync(importPath), 
            `Import file missing: ${exportConfig.import}`);
          assert.ok(existsSync(typesPath), 
            `Types file missing: ${exportConfig.types}`);
        }
      });
    });
  });
});
