import _ from 'lodash';
import { NODE_TYPES } from './shared/types.js';
import { createTree } from './shared/id3-algorithm.js';
/**
 * Decision Tree Algorithm
 * @module DecisionTree
 */
/**
 * Decision Tree class implementing ID3 algorithm
 */
class DecisionTree {
    static NODE_TYPES = NODE_TYPES;
    model;
    data = [];
    target;
    features;
    constructor(...args) {
        const numArgs = args.length;
        if (numArgs === 1) {
            this.import(args[0]);
        }
        else if (numArgs === 2) {
            const [target, features] = args;
            if (!target || typeof target !== 'string') {
                throw new Error('`target` argument is expected to be a String. Check documentation on usage');
            }
            if (!features || !Array.isArray(features)) {
                throw new Error('`features` argument is expected to be an Array<String>. Check documentation on usage');
            }
            this.target = target;
            this.features = features;
        }
        else if (numArgs === 3) {
            const [data, target, features] = args;
            const instance = new DecisionTree(target, features);
            instance.train(data);
            return instance;
        }
        else {
            throw new Error('Invalid arguments passed to constructor. Check documentation on usage');
        }
    }
    /**
     * Trains the decision tree with provided data
     * @param data - Array of training data objects
     */
    train(data) {
        if (!data || !Array.isArray(data)) {
            throw new Error('`data` argument is expected to be an Array<Object>. Check documentation on usage');
        }
        this.model = createTree(data, this.target, this.features);
    }
    /**
     * Predicts class for a given sample
     * @param sample - Sample data to predict
     * @returns Predicted class value
     */
    predict(sample) {
        let root = this.model;
        while (root.type !== NODE_TYPES.RESULT) {
            let attr = root.name;
            let sampleVal = sample[attr];
            let childNode = _.find(root.vals, function (node) {
                return node.name == sampleVal;
            });
            if (childNode) {
                root = childNode.child;
            }
            else {
                root = root.vals[0].child;
            }
        }
        return root.val;
    }
    /**
     * Evaluates prediction accuracy on samples
     * @param samples - Array of test samples
     * @returns Accuracy ratio (correct predictions / total predictions)
     */
    evaluate(samples) {
        let total = 0;
        let correct = 0;
        _.each(samples, (s) => {
            total++;
            let pred = this.predict(s);
            let actual = s[this.target];
            if (_.isEqual(pred, actual)) {
                correct++;
            }
        });
        return correct / total;
    }
    /**
     * Imports a previously saved model with the toJSON() method
     * @param json - JSON representation of the model
     */
    import(json) {
        const { model, data, target, features } = json;
        this.model = model;
        this.data = data;
        this.target = target;
        this.features = features;
    }
    /**
     * Returns JSON representation of trained model
     * @returns JSON object containing model data
     */
    toJSON() {
        const { data, target, features } = this;
        const model = this.model;
        return { model, data, target, features };
    }
}
// Export the DecisionTree class
export default DecisionTree;
//# sourceMappingURL=decision-tree.js.map