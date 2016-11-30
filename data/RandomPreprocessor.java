package ml.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class RandomPreprocessor extends DataPreprocessor {

	HashMap<Integer, String> usableFeatures;
	ArrayList<Integer> unwantedFeatures;
	
	public DataSet preprocessTest(DataSet test) {
		ArrayList<Example> examples = (ArrayList<Example>) test.getData().clone();
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
	public DataSet preprocessTrain(DataSet data, int n){
		Random r = new Random();
		ArrayList<Example> examples = (ArrayList<Example>) data.getData().clone();
		HashMap<Integer,String> featureMap = data.getFeatureMap();
		//eliminate n random features
		ArrayList<Integer> unselected = new ArrayList<Integer>();
		for (int i : data.getAllFeatureIndices()){
			unselected.add(i);
		}
		unwantedFeatures = new ArrayList<Integer>();
		usableFeatures = new HashMap<Integer,String>();
		int toDelete;
		int deleted;
		for (int i=0; i<n; i++){
			toDelete = r.nextInt(unselected.size());
			deleted = unselected.remove(toDelete);
			unwantedFeatures.add(deleted);
		}
		for (int i : unselected){
			usableFeatures.put(i, featureMap.get(i));
		}
		return returnCopyWithFeatures(examples, usableFeatures, unwantedFeatures);
	}
	
}
