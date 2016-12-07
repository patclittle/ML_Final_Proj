package ml.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.GradientDescentClassifier;
import ml.utils.TestExperiments;

/**
 * Class that implements the feature ablation study
 * preprocessing method, as presented in lecture.
 * 
 * Patrick Little and Molly Driscoll
 * CS158 Final Project
 * @author plittle
 *
 */
public class AblationPreprocessor extends DataPreprocessor {
	
	//Instance variables
	HashMap<Integer, String> usableFeatures;
	ArrayList<Integer> unwantedFeatures;
	Classifier c;
	
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
		//Make copy of examples
		ArrayList<Example> examples = (ArrayList<Example>) data.getData().clone();
		HashMap<Integer,String> featureMap = data.getFeatureMap();
		
		//Setup for ablation study
		ArrayList<Integer> unselected = new ArrayList<Integer>();
		for (int i : data.getAllFeatureIndices()){
			unselected.add(i);
		}
		unwantedFeatures = new ArrayList<Integer>();
		usableFeatures = new HashMap<Integer,String>();
		
		//Conduct ablation study for each feature, saving accuracies
		HashMap<Integer, Double> featureErrors = new HashMap<Integer, Double>();
		for(int i : data.getAllFeatureIndices()){
			featureErrors.put(i, conductAblationStudy(i,examples, data));
		}
		//Sort errors in descending order
		Collections.sort(unselected, new Comparator<Integer>(){
			public int compare(Integer first, Integer second){
				return -1*(featureErrors.get(first).compareTo(featureErrors.get(second)));
			}
		});
		
		//Remove n worst features
		int deleted;
		for (int i=0; i<n; i++){
			deleted = unselected.remove(0);
			unwantedFeatures.add(deleted);
		}
		for (int i : unselected){
			usableFeatures.put(i, featureMap.get(i));
		}
		
		//Return copy of dataset with n features removed
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
	/**
	 * Preprocess method that takes in a classifier to use for the ablation study.
	 * 
	 * @param data - the dataset to preprocess
	 * @param n - number of features to remove
	 * @param c - the classifier to use
	 * @return a copy of the data with n features removed
	 */
	public DataSet preprocessTrain(DataSet data, int n, Classifier c){
		setClassifier(c);
		return preprocessTrain(data, n);
	}
	
	/**
	 * Sets the classifier to use in the ablation study
	 * @param c - the classifier
	 */
	public void setClassifier(Classifier c){
		this.c = c;
	}
	
	/**
	 * Does the actual work of the ablation study: remove the given feature
	 * from the dataset, and return classification accuracy for this new data.
	 * 
	 * @param featureNum - index of the feature to remove
	 * @param examples - the examples to use in making a copy of the dataset
	 * @param data - the dataset to remove a feature from
	 * @return the accuracy of the classifier on the preprocessed dataset
	 */
	private double conductAblationStudy(int featureNum, ArrayList<Example> examples, DataSet data){
		if(c==null){
			c = new DecisionTreeClassifier();
		}
		ArrayList<Integer> unwanted = new ArrayList<Integer>();
		unwanted.add(featureNum);
		HashMap<Integer,String> usable = new HashMap<Integer,String>();
		for (int i : data.getFeatureMap().keySet()){
			if (i!=featureNum){
				usable.put(i, data.getFeatureMap().get(i));
			}
		}
		DataSet toTest = returnCopyWithFeatures(examples, usable, unwanted);
		return TestExperiments.averageList(TestExperiments.testNFold(new CrossValidationSet(toTest,10),c,10));
	}
	
}
