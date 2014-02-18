var _ = require('underscore');

//ID3 Decision Tree Algorithm
//main algorithm and prediction functions

function ID3(_s, target, features) {
	this.data = _s;
	this.target = target;
	this.features = features;
	this.model = createTree(_s, target, features);
}

ID3.prototype = {
	predict: function(sample) {
		var root = this.model;
		while (root.type !== "result") {
			var attr = root.name;
			var sampleVal = sample[attr];
			var childNode = _.detect(root.vals,function(x) { return x.name == sampleVal });			
			root = childNode.child;
		}

		return root.val;
	},

	evaluate: function(samples) {
		var instance = this;
		var target = this.target;

		var total = 0;
		var correct = 0;
		
		_.each(samples, function(s) {
			total++;
			var pred = instance.predict(s);
			var actual = s[target];
			if(pred == actual) {
				correct++;
			}
		});

		return correct / total;
	}
};

/**
 * @module ID3
 */
module.exports = ID3;

/**
 * Private API
 */

function createTree(_s, target, features) {
	var targets = _.unique(_.pluck(_s, target));
	if (targets.length == 1){
		// console.log("end node! "+targets[0]);
		return {type:"result", val: targets[0], name: targets[0],alias:targets[0]+randomTag() }; 
	}
	if(features.length == 0){
		// console.log("returning the most dominate feature!!!");
		var topTarget = mostCommon(targets);
		return {type:"result", val: topTarget, name: topTarget, alias: topTarget+randomTag()};
	}
	var bestFeature = maxGain(_s,target,features);
	var remainingFeatures = _.without(features,bestFeature);
	var possibleValues = _.unique(_.pluck(_s, bestFeature));
	var node = {name: bestFeature,alias: bestFeature+randomTag()};
	node.type = "feature";
	node.vals = _.map(possibleValues,function(v){
		// console.log("creating a branch for "+v);
		var _newS = _s.filter(function(x) {return x[bestFeature] == v});
		var child_node = {name:v,alias:v+randomTag(),type: "feature_value"};
		child_node.child =  createTree(_newS,target,remainingFeatures);
		return child_node;
	});
	
	return node;
}

function entropy(vals){
	var uniqueVals = _.unique(vals);
	var probs = uniqueVals.map(function(x){return prob(x,vals)});
	var logVals = probs.map(function(p){return -p*log2(p) });
	return logVals.reduce(function(a,b){return a+b},0);
}

function gain(_s,target,feature){
	var attrVals = _.unique(_.pluck(_s, feature));
	var setEntropy = entropy(_.pluck(_s, target));
	var setSize = _.size(_s);
	var entropies = attrVals.map(function(n){
		var subset = _s.filter(function(x){return x[feature] === n});
		return (subset.length/setSize)*entropy(_.pluck(subset,target));
	});
	var sumOfEntropies =  entropies.reduce(function(a,b){return a+b},0);
	return setEntropy - sumOfEntropies;
}

function maxGain(_s,target,features){
	return _.max(features,function(e){return gain(_s,target,e)});
}

function prob(val,vals){
	var instances = _.filter(vals,function(x) {return x === val}).length;
	var total = vals.length;
	return instances/total;
}

function log2(n){
	return Math.log(n)/Math.log(2);
}

function mostCommon(l) {
	return  _.sortBy(l,function(a){
		return count(a,l);
	}).reverse()[0];
}

function count(a, l) {
	return _.filter(l,function(b) { return b === a}).length
}

function randomTag() {
	return "_r"+Math.round(Math.random()*1000000).toString();
}
