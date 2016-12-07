package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ThreadLocalRandom;

import ml.data.DataSet;
import ml.data.Example;

/**
 * Class that implements the Two Layer Neural network,
 * in accordance to class and homework notes.
 * 
 *  Patrick Little and Molly Driscoll
 * CS158 Final Project
 * @author plittle
 *
 */
public class TwoLayerNN implements Classifier {

	//Instance variables that can be set from outside
	private int numHidden;
	private double eta = 0.1;
	private int iterations = 200;
	
	//Instance variables hidden from outside
	private DataSet theData;
	private double[][] inputWeights;
	private double[] innerWeights;
	
	/**
	 * Constructor for two-layer neural network
	 * @param numHidden - number of nodes in hidden layer
	 */
	public TwoLayerNN(int numHidden){
		this.numHidden = numHidden;
	}
	
	/**
	 * Trains the neural network, in accordance with the lecture and homework notes.
	 * 
	 * @param data - the dataset to train on
	 */
	@Override
	public void train(DataSet data) {
		//Initialize weights and bias
		setupForTraining(data);
		ArrayList<Example> examples = theData.getData();
		for (int i=0; i<iterations; i++){
			trainIteration(examples);
		}
	}
	
	/**
	 * Sets up the initial weights of network, and adds a bias to the data
	 * 
	 * @param data - the data to train on
	 */
	private void setupForTraining(DataSet data){
		//Add bias to dataset
		theData = data.getCopyWithBias();
		//Initialize weights
		initializeWeights();
	}
	
	/**
	 * Does a single training iteration, as specified by lecture and homework notes.
	 * 
	 * @param examples - the examples to train on
	 */
	private void trainIteration(ArrayList<Example> examples){
		//Initialize local variables
		double[] innerOutputs;
		double[] innerDerivatives;
		double[] innerActivations;
		double output;
		double outputActivation;
		double outputDerivative;
		double[] innerWeightUpdates;
		double[][] inputWeightUpdates;
		
		Collections.shuffle(examples);
		for (Example ex : examples){
			//Calculate outputs (and save activation derivatives), going forward
			innerActivations = calculateInnerActivations(ex);
			innerOutputs = calculateInnerOutputs(innerActivations);
			innerDerivatives = calculateInnerDerivatives(innerActivations);
			outputActivation = calculateOutputActivation(innerOutputs);
			output = Math.tanh(outputActivation);
			outputDerivative = 1-Math.pow(Math.tanh(outputActivation),2);
			
			//Calculate new weights for output layer
			innerWeightUpdates = calculateInnerWeightUpdates(innerOutputs,outputDerivative,output,ex.getLabel());
			
			//Calculate new weights for hidden layer
			inputWeightUpdates = calculateInputWeightUpdates(ex,innerDerivatives,outputDerivative,output);
			
			//Update model with new weights
			innerWeights = innerWeightUpdates.clone();
			inputWeights = inputWeightUpdates.clone();
		}
		
	}

	/**
	 * Classifies an example using the trained network
	 * 
	 * @param example - the example to classify
	 */
	@Override
	public double classify(Example example) {
		return classifyOrConfidence(example, true);
	}

	/**
	 * Return the confidence of our prediction, calculated 
	 * based on the absolute value of the output of the network.
	 * 
	 * @param example - the example to calculate confidence for
	 */
	@Override
	public double confidence(Example example) {
		return classifyOrConfidence(example,false);
	}
	
	/**
	 * Depending on the flag, will either return the classification
	 * or confidence for the example, based on the algorithm in the
	 * slides and homework notes.
	 * 
	 * @param example - example to calculate output/confidence for
	 * @param classify - if true, output classification, else output confidence
	 * @return classification or confidence, depending on classify flag
	 */
	private double classifyOrConfidence(Example example, boolean classify){
		example = theData.addBiasFeature(example);
		double[] innerOutputs = calculateInnerOutputs(calculateInnerActivations(example));
		double output = Math.tanh(calculateOutputActivation(innerOutputs));
		if (classify){
			return output > 0 ? 1.0 : -1.0;
		}else{
			return Math.abs(output);
		}
	}
	
	/**
	 * Sets the learning rate for the training method
	 * 
	 * @param newEta - the learning rate to set
	 */
	public void setEta(double newEta){
		eta = newEta;
	}

	/**
	 * Sets the number of times that the training method 
	 * should iterate through all examples.
	 * 
	 * @param its - the number of iterations to set
	 */
	public void setIterations(int its){
		iterations = its;
	}
	
	/**
	 * Initializes the weights in the neural network to have random
	 * weights between -1 and 1.
	 */
	private void initializeWeights(){
		//Initialize input weights
		inputWeights = new double[numHidden][theData.getAllFeatureIndices().size()];
		for (int j=0; j<numHidden; j++){
			for (int k=0; k<theData.getAllFeatureIndices().size(); k++){
				inputWeights[j][k] = ThreadLocalRandom.current().nextDouble(-1.0, 1.0);
			}
		}
		//Initialize weights from hidden nodes to output node
		innerWeights = new double[numHidden+1];
		for (int i=0; i<innerWeights.length; i++){
			innerWeights[i] = ThreadLocalRandom.current().nextDouble(-1.0, 1.0);
		}
	}
	
	/**
	 * Calculates input to the activation functions of all hidden nodes, given an example.
	 * 
	 * @param ex - the example to calculate output for
	 * @return a list with the inputs the the activation functions of all hidden nodes
	 */
	private double[] calculateInnerActivations(Example ex){
		double activationInput;
		double[] weightList;
		double[] toReturn = new double[numHidden];
		//For each set of weigths associated with each node
		for (int i = 0; i<inputWeights.length; i++){
			activationInput = 0;
			weightList = inputWeights[i];
			//calculate dot product of weights * feature values
			for (int j=0; j<weightList.length; j++){
				activationInput += weightList[j]*ex.getFeature(j);
			}
			//calculate activation function output
			toReturn[i] = activationInput;
		}
		return toReturn;
	}
	
	/**
	 * Calculates the outputs of the activation functions of the nodes in the hidden layer
	 * 
	 * @param inputs - the inputs to the activation function (for each node: cross product of the features
	 * and the weights associated with the node)
	 * 
	 * @return a list of the outputs for nodes in the hidden layer
	 */
	public double[] calculateInnerOutputs(double[] inputs){
		double[] toReturn = new double[inputs.length];
		for (int i=0; i<inputs.length; i++){
			toReturn[i] = Math.tanh(inputs[i]);
		}
		return toReturn;
	}
	
	/**
	 * Calculates the derivative of the activation function
	 * for nodes in the hidden layer.
	 * 
	 * @param inputs - the inputs to the activation function (for each node: cross product of the features
	 * and the weights associated with the node)
	 * @return a list of the derivative of the activation function for nodes in the hidden layer.
	 */
	public double[] calculateInnerDerivatives(double[] inputs){
		double[] toReturn = new double[inputs.length];
		for (int i=0; i<inputs.length; i++){
			toReturn[i] = 1-Math.pow(Math.tanh(inputs[i]),2);
		}
		return toReturn;
	}
	
	/**
	 * Calculates the input to the activation function of the
	 * output node, given the hidden layer outputs.
	 * 
	 * @param innerOutputs - the outputs of the inner nodes
	 * @return the input to the output activation function
	 */
	private double calculateOutputActivation(double[] innerOutputs){
		//Initiliaze as 1*finalWeight to account for bias
		double activationInput = innerWeights[innerWeights.length-1];
		for (int i=0; i<innerOutputs.length; i++){
			activationInput += innerOutputs[i]*innerWeights[i];
		}
		return activationInput;
	}
	
	/**
	 * Calculates the new weights for the edges between hidden nodes
	 * and the output nodes, based on the update function in the slides.
	 * 
	 * @param innerOutputs - the outputs from the hidden nodes
	 * @param outputDerivative - the derivative of the activation function of the 
	 * output node
	 * @param output - the output of the output node
	 * @param label - the label of the example
	 * @return a list of new weights for the edges between hidden nodes and the output node.
	 */
	private double[] calculateInnerWeightUpdates(double[] innerOutputs,double outputDerivative,double output,double label){
		double[] toReturn = innerWeights.clone();
		for (int i = 0; i<toReturn.length-1; i++){
			toReturn[i]+=eta*innerOutputs[i]*(label-output)*outputDerivative;
		}
		//To account for bias
		toReturn[toReturn.length-1]+=eta*(label-output)*outputDerivative;
		return toReturn;
	}
	
	/**
	 * Calculates the new weights for the edges between the inputs
	 * and the hidden nodes, based on the update function in the slides.
	 * 
	 * @param ex - the example to update based on
	 * @param innerDerivatives - the derivatives of the activation function for all
	 * inner nodes.
	 * @param outputDerivative - the derivative of the activation function for the output node
	 * @param output - the output of the output node
	 * @return the new weights for the edges between the inputs
	 * and the hidden nodes
	 */
	private double[][] calculateInputWeightUpdates(Example ex,double[] innerDerivatives,double outputDerivative,double output){
		double[][] toReturn = inputWeights.clone();
		double[] weightList;
		for (int j = 0; j<toReturn.length; j++){
			weightList = toReturn[j];
			for (int k=0; k<weightList.length; k++){
				weightList[k]+=eta*ex.getFeature(k)*innerDerivatives[j]*innerWeights[j]*outputDerivative*(ex.getLabel()-output);
			}
		}
		return toReturn;
	}

}
