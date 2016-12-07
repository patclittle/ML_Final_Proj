package ml.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * A class that implements a random feature preprocessor, which
 * removes n random features from a dataset.
 * 
 * Patrick Little and Molly Driscoll
 * CS158 Final Project
 * 
 * @author plittle
 *
 */
public class RandomPreprocessor extends DataPreprocessor {

	//instance variables
	HashMap<Integer, String> usableFeatures;
	ArrayList<Integer> unwantedFeatures;
	
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
	 * as decided by random selection
	 * 
	 * @param data - the dataset to preprocess
	 * @param n - number of features to remove
	 */
	public DataSet preprocessTrain(DataSet data, int n){
		Random r = new Random();
		ArrayList<Example> examples = (ArrayList<Example>) data.getData().clone();
		HashMap<Integer,String> featureMap = data.getFeatureMap();
		//Setup for elimination
		ArrayList<Integer> unselected = new ArrayList<Integer>();
		for (int i : data.getAllFeatureIndices()){
			unselected.add(i);
		}
		unwantedFeatures = new ArrayList<Integer>();
		usableFeatures = new HashMap<Integer,String>();
		int toDelete;
		int deleted;
		//Eliminate n random features
		for (int i=0; i<n; i++){
			toDelete = r.nextInt(unselected.size());
			deleted = unselected.remove(toDelete);
			unwantedFeatures.add(deleted);
		}
		for (int i : unselected){
			usableFeatures.put(i, featureMap.get(i));
		}
		//Return copy of dataset with features removed
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
}
