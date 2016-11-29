package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;


/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * Patrick Little, Assignment 7
 * 
 * @author dkauchak
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;
	
	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;
	
	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight
	
	protected int iterations = 10;
	
	//INstance variable that are used to customize loss and regularization
	protected int loss = 2;
	protected int regularization = 0;
	
	//Instance variables for the hyperparameters lambda and eta
	protected double lambda = 0.1;
	protected double eta = 0.1;
		
	/**
	 * Get a weight vector over the set of features with each weight
	 * set to 0
	 * 
	 * @param features the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features){
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();
		
		for( Integer f: features){
			temp.put(f, 0.0);
		}
		
		return temp;
	}
	
	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features){
		weights = getZeroWeights(features);
		b = 0;
	}
	
	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations){
		this.iterations = iterations;
	}
	
	/**
	 * Sets the loss function according to the constants
	 * 
	 * @param newLoss - loss function to choose
	 */
	public void setLoss(int newLoss){
		if (newLoss == EXPONENTIAL_LOSS || newLoss == HINGE_LOSS){
			loss = newLoss;
		}
	}
	
	
	/**
	 * Sets the regularization method based on the constants
	 * 
	 * @param newReg - regularization method to choose
	 */
	public void setRegularization(int newReg){
		if(newReg == NO_REGULARIZATION || newReg == L1_REGULARIZATION || newReg == L2_REGULARIZATION){
			regularization = newReg;
		}
	}
	
	/**
	 * Sets the value for lambda, the regularization constant
	 * 
	 * @param newLambda - lambda value to set
	 */
	public void setLambda(double newLambda){
		lambda = newLambda;
	}
	
	/**
	 * Sets the value for eta, the step constant
	 * 
	 * @param newEta
	 */
	public void setEta(double newEta){
		eta = newEta;
	}
	
	/**
	 * Trains the gradient descent classifier on a data set,
	 * in accordance to the lecture notes
	 * 
	 * @param data - the set to train on
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());
		
		ArrayList<Example> training = (ArrayList<Example>)data.getData().clone();
		
		for( int it = 0; it < iterations; it++ ){
			Collections.shuffle(training);
			
			for( Example e: training ){

				double label = e.getLabel();
				double prediction = getPrediction(e);
				double distance = getDistanceFromHyperplane(e,weights,b);
				
				// update the weights
				//for( Integer featureIndex: weights.keySet() ){
				for( Integer featureIndex: e.getFeatureSet() ){
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);
					weights.put(featureIndex, oldWeight + (eta*(getLossCorrection(label, 
							distance, featureValue) - getRegularizationCorrection(oldWeight))));
				}
				
					
				// update b
				b += (eta*(getLossCorrection(label, 
						distance, 1) - getRegularizationCorrection(b)));
				
			}
		}
	}
	
	/**
	 * Gets the correction to a weight based on the regularization method.
	 * If there is no regularization, there is no correction. Else, the
	 * regularization amount is calcualted according to the slides.
	 * 
	 * @param oldWeight - the old weight to regularize
	 * @return the amount to regularize by
	 */
	private double getRegularizationCorrection(double oldWeight){
		double toReturn = lambda; //L1 and L2 will both use lambda
		if (regularization == NO_REGULARIZATION){
			return 0.0;
		}else if (regularization == L1_REGULARIZATION){
			//return lambda*sign(weight)
			if (oldWeight < 0){
				toReturn*=-1;
			}else{
				toReturn*=1;
			}
		}else{
			//return lambda*weight
			toReturn*=oldWeight;
		}
		return toReturn;
	}
	
	/**
	 * Gets the loss correction amount (depending on the loss function).
	 * That is, return featureValue*label*c, where c depends on the
	 * loss function.
	 * 
	 * @param label - label of example
	 * @param prediction - prediction for example (used for hinge)
	 * @param distance - distance of example from hyperplane (used for exponential)
	 * @param featureValue - value corresponding to the weight
	 * @return the loss correction
	 */
	private double getLossCorrection(double label, double distance, double featureValue){
		double toReturn = label*featureValue;
		if (loss == EXPONENTIAL_LOSS){
			toReturn *= Math.exp(-1*label*distance);
		}else{
			if (distance*label<1){
				toReturn*=1;
			}else{
				toReturn*=0;
			}
		}
		return toReturn;
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}
	
	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

		
	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e the example to predict
	 * @return
	 */
	protected double getPrediction(Example e){
		return getPrediction(e, weights, b);
	}
	
	/**
	 * Get the prediction from the on this example from using weights w and inputB
	 * 
	 * @param e example to predict
	 * @param w the set of weights to use
	 * @param inputB the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = getDistanceFromHyperplane(e,w,inputB);

		if( sum > 0 ){
			return 1.0;
		}else if( sum < 0 ){
			return -1.0;
		}else{
			return 0;
		}
	}
	
	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB){
		double sum = inputB;
		
		//for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for( Integer featureIndex: e.getFeatureSet()){
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}
		
		return sum;
	}
	
	public String toString(){
		StringBuffer buffer = new StringBuffer();
		
		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);
		
		for(Integer index: temp){
			buffer.append(index + ":" + weights.get(index) + " ");
		}
		
		return buffer.substring(0, buffer.length()-1);
	}
}
