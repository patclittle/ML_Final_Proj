package ml.utils;

import java.util.ArrayList;
import java.util.Random;

import ml.classifiers.Classifier;
import ml.classifiers.DecisionTreeClassifier;
import ml.classifiers.GradientDescentClassifier;
import ml.classifiers.KNNClassifier;
import ml.classifiers.TwoLayerNN;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Class for the experimentation for the homework.
 * 
 * Patrick Little, Assignment 8
 * 
 * @author plittle
 *
 */
public class TestExperiments {
	
	public static void main(String[] args) {
		//TO-DO: Read in data
		ArrayList<DataSet> datasets = new ArrayList<DataSet>();
		//TO-DO: uncomment the stuff from before
		ArrayList<Classifier> cs = new ArrayList<Classifier>();
		cs.add(new DecisionTreeClassifier());
		cs.add(new GradientDescentClassifier());
		cs.add(new KNNClassifier());
		cs.add(new TwoLayerNN(7));
		
		/**ArrayList<PreProcessor> pps = new ArrayList<PreProcessor>();
		pps.add(MOLLIES);
		pps.add(RANDOM);
		
		for (DataSet data : datasets){
			printTablesForClassifiers(pps,cs,data,50,10,10);
		}
		*/
		
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
	private static double[] testNFold(CrossValidationSet cvs, Classifier c, int nFolds){
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
	private static double averageList(double[] toAverage){
		double toReturn = 0.0;
		for (double d : toAverage){
			toReturn+=d;
		}
		return toReturn/toAverage.length;
	}
	
	private static double getAccuracyForClassifier(Classifier c, DataSet data, int iterationsToAvg, int nFolds){
		double[] nFoldAvgs = new double[iterationsToAvg];
		CrossValidationSet cvs = new CrossValidationSet(data, nFolds);
		for (int i=0; i<iterationsToAvg; i++){
			nFoldAvgs[i] = averageList(testNFold(cvs,c,nFolds));
		}
		return averageList(nFoldAvgs);
	}
	
	//private static double[] getNAccuraciesForPreProcessor(PreProcessor pp, Classifier c, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
	private static double[] getNAccuraciesForPreProcessor(Classifier c, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		double[] toReturn = new double[nFeatures];
		toReturn[0] = getAccuracyForClassifier(c,data,iterationsToAvg,nFolds);
		for (int i=1; i<nFeatures; i++){
			//data = pp.preProcessData(data,i);
			toReturn[i] = getAccuracyForClassifier(c,data,iterationsToAvg,nFolds);
		}
		return toReturn;
	}
	
	/**
	private static void printTableForPreProcessors(PreProcessor[] pps, Classifier c, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		for (PreProcessor pp : pps){
			printRowFromList(getNAccuraciesForPreProcessor(pp,c,data,iterationsToAvg,nFolds,nFeatures));
		}
	}*/
	
	/**
	private static void printTablesForClassifiers(PreProcessor[] pps, Classifier[] cs, DataSet data, int iterationsToAvg, int nFolds, int nFeatures){
		for (Classifier c : cs){
			printTableForPreProcessors(pps,c,data,iterationsToAvg,nFolds,nFeatures);
		}
	}*/
	
	private static void printRowFromList(double[] toPrint){
		System.out.print(toPrint[0]);
		for (int i = 1; i<toPrint.length; i++){
			System.out.print(","+toPrint[i]);
		}
		System.out.println();
	}
}