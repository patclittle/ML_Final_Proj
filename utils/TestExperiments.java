package ml.utils;

import java.util.ArrayList;
import java.util.Random;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.GradientDescentClassifier;
import ml.classifiers.KNNClassifier;
import ml.classifiers.TwoLayerNN;
import ml.data.AblationPreprocessor;
import ml.data.CrossValidationSet;
import ml.data.DTpreprocessor;
import ml.data.DataPreprocessor;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.data.RandomPreprocessor;

/**
 * Class for testing and evaluating preprocessors
 * 
 * Patrick Little and Molly Driscoll
 * CS158 Final Project
 * 
 * @author plittle
 *
 */
public class TestExperiments {
	
	/**
	 * Main method which makes a table for the accuracies
	 * of all classifiers and preprocessors for a range
	 * of # of preprocessed features.
	 * 
	 * @param args - takes no args
	 */
	public static void main(String[] args) {
		//Setup datasets
		ArrayList<DataSet> datasets = new ArrayList<DataSet>();
		datasets.add(new DataSet("/Users/plittle/workspace/ML_Final_Proj/src/kr-vs-kp.data.csv",DataSet.CSVFILE));
		
		//Setup classifiers
		ArrayList<Classifier> cs = new ArrayList<Classifier>();
		cs.add(new DecisionTreeClassifier());
		cs.add(new GradientDescentClassifier());
		cs.add(new KNNClassifier());
		cs.add(new TwoLayerNN(7));
		
		//Setup preprocessors
		ArrayList<DataPreprocessor> pps = new ArrayList<DataPreprocessor>();
		pps.add(new RandomPreprocessor());
		pps.add(new DTpreprocessor());
		pps.add(new AblationPreprocessor());
		
		//Print tables
		for (DataSet data : datasets){
			printTablesForClassifiers(pps,cs,data,1,10,6);
		}	
	}
	
	/**
	 * Test a classifier on a data split, returning the accuracy
	 * 
	 * @param c - the classifier to test
	 * @param split - the split to train and test on
	 * @return the accuracy of the classifier
	 */
	private static double testClassifierOnSplit(Classifier c, DataSetSplit split){
		double totalCorrect = 0.0;
		c.train(split.getTrain());
		for (Example e : split.getTest().getData()){
			if (e.getLabel() == c.classify(e)){
				totalCorrect +=1;
			}
		}
		return totalCorrect/split.getTest().getData().size();
	}
	
	/**
	 * Tests a classifier on n folds on a given data set
	 * 
	 * @param cvs - the CrossValidationSet to draw from
	 * @param c - the classifier to train
	 * @param nFolds - the number of folds to train and test on
	 * @return a list of the accuracy of the classifier on each split
	 */
	public static double[] testNFold(CrossValidationSet cvs, Classifier c, int nFolds){
		double[] toReturn = new double[nFolds];
		for (int i = 0; i<nFolds; i++){
			DataSetSplit split = cvs.getValidationSet(i);
			toReturn[i] = testClassifierOnSplit(c, split);
		}
		return toReturn;
	}
	
	/**
	 * Returns the average of a list of doubles
	 * 
	 * @param toAverage - the list to average
	 * @return the average of the list
	 */
	public static double averageList(double[] toAverage){
		double toReturn = 0.0;
		for (double d : toAverage){
			toReturn+=d;
		}
		return toReturn/toAverage.length;
	}
	
	/**
	 * Returns the average n-fold cross validation accuracy
	 * for a classifier on a dataset, averaged over some number
	 * of iterations.
	 * 
	 * @param c - the classifier
	 * @param data - the dataset to train and test on
	 * @param iterationsToAvg - number of n-folds to average
	 * @param nFolds - number of folds per n-fold validation 
	 * @return the average accuracy of the classifier over all cross validations
	 */
	private static double getAccuracyForClassifier(Classifier c, DataSet data, int iterationsToAvg, int nFolds){
		double[] nFoldAvgs = new double[iterationsToAvg];
		CrossValidationSet cvs = new CrossValidationSet(data, nFolds);
		for (int i=0; i<iterationsToAvg; i++){
			nFoldAvgs[i] = averageList(testNFold(cvs,c,nFolds));
		}
		return averageList(nFoldAvgs);
	}
	
	/**
	 * For a range of values for n, this method will preprocess n features
	 * and remove them from the dataset, and return a list of the accuracies
	 * of the classifier on the preprocessed dataset.
	 * 
	 * @param pp - the preprocessor to use
	 * @param c - the classifier
	 * @param data - the dataset to preprocess
	 * @param iterationsToAvg - how many iterations to average the accuracy of
	 * @param nFolds - how many folds per n-fold cross validations
	 * @param nFeatures - how many features (maximum) will be preprocessed
	 * @return
	 */
	private static double[] getNAccuraciesForPreProcessor(DataPreprocessor pp, Classifier c, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		double[] toReturn = new double[nFeatures];
		//Get initial accuracy before preprocessing
		toReturn[0] = getAccuracyForClassifier(c,data,iterationsToAvg,nFolds);
		//Note: ablation preprocessor will use the same classifier to
		// preprocess as we use in our evaluation
		if(pp.getClass()== AblationPreprocessor.class){
			((AblationPreprocessor) pp).setClassifier(c);
		}
		//Collect preprocessing accuracies
		DataSet newData = pp.preprocessTrain(data, 1);
		for (int i=1; i<nFeatures; i++){
			System.out.println(newData.getAllFeatureIndices().size());
			toReturn[i] = getAccuracyForClassifier(c,newData,iterationsToAvg,nFolds);
			newData = pp.preprocessTrain(newData,1);
		}
		return toReturn;
	}
	
	/**
	 * Prints the accuracies for a range of n preprocessed features for 
	 * each preprocessor passed to the function.
	 * 
	 * @param pps - the preprocessors to print for
	 * @param c - the classifier to use for evaluation
	 * @param data - the dataset to preprocess
	 * @param iterationsToAvg - how many iterations to average the accuracy of
	 * @param nFolds - how many folds in the n-fold cross validation
	 * @param nFeatures - how many features (maximum) will be preprocessed
	 */
	private static void printTableForPreProcessors(ArrayList<DataPreprocessor> pps, Classifier c, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		for (DataPreprocessor pp : pps){
			printRowFromList(getNAccuraciesForPreProcessor(pp,c,data,iterationsToAvg,nFolds,nFeatures));
		}
	}
	
	/**
	 * Given a list of classifiers and preprocessors, for each possible pairing,
	 * prints the classification accuracy for a range of n preprocessed features.
	 * 
	 * @param pps - list of preprocessors
	 * @param cs - list of classifiers
	 * @param data - dataset to preprocess
	 * @param iterationsToAvg - iterations to average the accuracy of
	 * @param nFolds - folds in the n-fold cross-validations
	 * @param nFeatures - number of features (maximum) to remove
	 */
	private static void printTablesForClassifiers(ArrayList<DataPreprocessor> pps, ArrayList<Classifier> cs, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		for (Classifier c : cs){
			printTableForPreProcessors(pps,c,data,iterationsToAvg,nFolds,nFeatures);
		}
	}
	
	/**
	 * Prints a list of doubles
	 * 
	 * @param toPrint - the list to print
	 */
	private static void printRowFromList(double[] toPrint){
		System.out.print(toPrint[0]);
		for (int i = 1; i<toPrint.length; i++){
			System.out.print(","+toPrint[i]);
		}
		System.out.println();
	}
}
