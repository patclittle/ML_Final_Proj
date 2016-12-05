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

public class AblationPreprocessor extends DataPreprocessor {
	
	HashMap<Integer, String> usableFeatures;
	ArrayList<Integer> unwantedFeatures;
	Classifier c;
	
	public DataSet preprocessTest(DataSet test) {
		ArrayList<Example> examples = (ArrayList<Example>) test.getData().clone();
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
	public DataSet preprocessTrain(DataSet data, int n){
		ArrayList<Example> examples = (ArrayList<Example>) data.getData().clone();
		HashMap<Integer,String> featureMap = data.getFeatureMap();
		
		ArrayList<Integer> unselected = new ArrayList<Integer>();
		for (int i : data.getAllFeatureIndices()){
			unselected.add(i);
		}
		unwantedFeatures = new ArrayList<Integer>();
		usableFeatures = new HashMap<Integer,String>();
		
		//Setup for ablation study
		HashMap<Integer, Double> featureErrors = new HashMap<Integer, Double>();
		for(int i : data.getAllFeatureIndices()){
			featureErrors.put(i, conductAblationStudy(i,examples, data));
		}
		Collections.sort(unselected, new Comparator<Integer>(){
			public int compare(Integer first, Integer second){
				return -1*(featureErrors.get(first).compareTo(featureErrors.get(second)));
			}
		});
		int deleted;
		for (int i=0; i<n; i++){
			deleted = unselected.remove(0);
			unwantedFeatures.add(deleted);
		}
		for (int i : unselected){
			usableFeatures.put(i, featureMap.get(i));
		}
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
	public DataSet preprocessTrain(DataSet data, int n, Classifier c){
		setClassifier(c);
		return preprocessTrain(data, n);
	}
	
	public void setClassifier(Classifier c){
		this.c = c;
	}
	
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
