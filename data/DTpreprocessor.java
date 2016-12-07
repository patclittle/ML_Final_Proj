package ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;
import java.util.AbstractMap;
import java.util.Map;


import ml.data.Example;
import ml.classifiers.DecisionTreeNode;
import ml.data.DataPreprocessor;
import ml.data.DataSet;
import ml.utils.HashMapCounter;

/**
 * Implements the training error feature selection (decision tree)
 * preprocessor, as described in lecture.
 * 
 *  Patrick Little and Molly Driscoll
 * CS158 Final Project
 * 
 * @author plittle
 *
 */
public class DTpreprocessor extends DataPreprocessor{
	HashMap<Integer, String> usableFeatures;
	ArrayList<Integer> unwantedFeatures;

	public DTpreprocessor(){
		
	}
	
	
	/**
	 * Preprocess test data, based on features removed in training
	 * 
	 * @param test - the dataset to preprocess
	 * @return a copy of the dataset with features removed
	 */
	public DataSet preprocessTest(DataSet test) {
		ArrayList<Example> examples = (ArrayList<Example>) test.getData().clone();
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
	/**
	 * Preprocess training data, by removing the n worst features
	 * as decided by a feature ablation study.
	 * 
	 * @param data - the dataset to preprocess
	 * @param n - number of features to remove
	 */
	public DataSet preprocessTrain(DataSet data, int n){
		ArrayList<Example> examples = (ArrayList<Example>) data.getData().clone();
		Set<Integer> features = data.getAllFeatureIndices();
		if(n >= features.size() + 1){
			throw new RuntimeException("Trying to eliminate too many features");
		}
		//Collect feature errors, as decided by a decision tree
		ArrayList<Map.Entry<Integer, Double>> featureErrors = new ArrayList<Map.Entry<Integer, Double>>();		
		for(Integer f:features){
			featureErrors.add(new AbstractMap.SimpleEntry<Integer, Double>(f, averageTrainingError(examples,f)));
		}
		Collections.sort(featureErrors, new Comparator<Map.Entry<Integer, Double>>(){
			public int compare(Map.Entry<Integer, Double> e1, Map.Entry<Integer, Double> e2){
				return -e1.getValue().compareTo(e2.getValue());
			}
		});
		//eliminate the n worst features
		unwantedFeatures = new ArrayList<Integer>();
		usableFeatures = new HashMap<Integer, String>();

		for(int i = 0; i < n; i ++){
			unwantedFeatures.add(featureErrors.remove(0).getKey());
		}
		for(Map.Entry<Integer, Double> f:featureErrors){
			Integer key = f.getKey();
			usableFeatures.put(key, key.toString());
		}
		//Return a copy of the dataset with worst features removed
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);

	}
	
	/**
	 * Returns the average training error if only the specified feature were used
	 * to predict the data.
	 * 
	 * @param data - the list of examples to calculate training error for
	 * @param featureIndex - the feature to split on
	 * @return the average training error after splitting on the feature
	 */
	private double averageTrainingError(ArrayList<Example> data, int featureIndex){		
		ArrayList<Example>[] splits = splitData(data, featureIndex);
		
		int leftCount = splits[0].size() > 0 ? getMajorityLabel(splits[0]).majorityCount : 0;
		int rightCount = splits[1].size() > 0 ? getMajorityLabel(splits[1]).majorityCount : 0;
		
		double accuracy = (leftCount+rightCount)/(double)data.size();
		return 1-accuracy;
	}
	
	/**
	 * Calculates the majority label for a set of examples
	 * 
	 * @param data - the set of examples
	 * @return the majority label
	 */
	private DataMajority getMajorityLabel(ArrayList<Example> data){
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		
		for( Example d: data ){
			counter.increment(d.getLabel());
		}
		
		double maxLabel = 0.0;
		int maxCount = -1;
		
		//Count majority label and count
		for( Double key: counter.keySet() ){
			if( counter.get(key) > maxCount ){
				maxCount = counter.get(key);
				maxLabel = key;
			}
		}
		
		//Return majority information (label, count, confidence)
		return new DataMajority(maxLabel, maxCount, ((double)maxCount)/data.size());
	}
	
	/**
	 * Split the data based on featureIndex
	 * 
	 * @param data the data to be split
	 * @param featureIndex the feature to split on
	 * @return the split of the data.  Entry 0 is the left branch data and entry 1 the right branch data.
	 */
	private ArrayList<Example>[] splitData(ArrayList<Example> data, int featureIndex){
		ArrayList<Example>[] splits = new ArrayList[2];
		splits[0] = new ArrayList<Example>();
		splits[1] = new ArrayList<Example>();
		
		// split the data based on this feature
		for( Example d: data){
			double value = d.getFeature(featureIndex);
			
			if( value == DecisionTreeNode.LEFT_BRANCH ){
				splits[0].add(d);
			}else{
				splits[1].add(d);
			}
		}
		
		return splits;
	}
	
	/**
	 * Class for representing information about the majority labels of a set of data
	 * 
	 * @author plittle
	 *
	 */
	private class DataMajority{
		
		//Stored information
		public double majorityLabel;
		public int majorityCount;
		public double confidence;
		
		public DataMajority(double majorityLabel, int majorityCount, double confidence){
			this.majorityLabel = majorityLabel;
			this.majorityCount = majorityCount;
			this.confidence = confidence;
		}
	}



}
