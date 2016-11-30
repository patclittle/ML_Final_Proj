package ml.data;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Interface defining the data preprocessing
 * 
 * @author dkauchak
 *
 */
public class DataPreprocessor {
	/**
	 * Preprocess the training data
	 * 
	 * @param train
	 */
	public DataSet preprocessTrain(DataSet train, int n){
		return null;
	}
	
	/**
	 * Preprocess the testing data
	 * 
	 * @param test
	 */
	public DataSet preprocessTest(DataSet test){
		return null;
	}
	
	
	protected DataSet returnCopyWithFeatures(ArrayList<Example> examples, HashMap<Integer,String> usableFeatures, ArrayList<Integer> unwantedFeatures){
		DataSet newData = new DataSet(usableFeatures);
		//remove unwanted features
		Example newExample = null;
		for(Example e:examples){
			newExample = new Example(e);
			for(Integer f: unwantedFeatures){
				newExample.getFeatureSet().remove(f);
			}
			newData.addData(newExample);
		}
		return newData;
	}
	
}