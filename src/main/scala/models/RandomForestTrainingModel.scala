package models

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

/*
 * Author 1: Priyanka Bhadravathi Halesh 
 * Author 2: Vishal Sanjiv Kotak
 * The below Spark program experiments with the working of Random Forest Model
 * creation from the Spark MLLib libraries in Scala. The input .csv file was
 * converted to LibSVM Format (DenseVectors, LabelPoint) acting as an argument
 * for the training model functionality of Random Forest. DenseVectors is a
 * vector of all the features for a particular record (21 * 21 * 7 = 3087). 
 * LabelPoint is the true label of the pixel i.e. it can be either foreground
 * or background. (Foreground represents 1, background represents 0). Various 
 * features like number of decision trees, maximum allowed depth of each 
 * decision tree has been varied and the resulting values were considered 
 * taking the Accuracy into account. Since, it is a classification problem,
 * we have used the method of train classifier from Random Forest packages.
 * The input training path and the saved model location are given as 
 * arguments.
 * 
 */

object RandomForestTrainingModel {
  
   def main(args: Array[String]): Unit = {
    
    // Creating the Spark Configuration object
    val conf = new SparkConf()
                  .setAppName("RandomForestTraining")
                  .setMaster(args(0));
    
    // Creating the Spark Context object
    val sc = new SparkContext(conf);
    
    // Loading the training data which has been stores as LibSVM format
    // (DenseVector, LabelPoint): DenseVector represents the 3087 
    // neighborhood brightness pixel values and LabelPoint is the 
    // pixel classification
    val trainingData = MLUtils.loadLibSVMFile(sc, args(1));
    
    // Number of classes to be predicted 
    // 2 for Foreground and Background
    
    val numClasses = 2
    
    // This represents the presence of categorical data in feature set
    // Since, the feature set does not contain categorical values, we 
    // will consider the value to be empty.
    val categoricalFeaturesInfo = Map[Int, Int]();
    
    // Number of trees to be created for every training record as a 
    // part of the Ensemble.
    val numTrees = 12;
    
    // Different features can be provided to different decision trees 
    // belonging to same forest. "auto" has been enabled for the 
    // Random forest to follow a greedy approach in selection of
    // attributes to form decision trees.
    val featureSubsetStrategy = "auto";
    
    // This specifies the presence of classification/regression problem
    val impurity = "gini";
    
    // This indicates the maximum depth of all the decision trees created
    // as a part of Random forest
    val maxDepth = 6
    
    // This parameter is used to discretizing categorical data and does 
    // not provide valuable significance. Hence this value has been set
    // to the default value.
    val maxBins = 32

    // Applying the above parameters to train the Random Forest model on the given training data
    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
                                  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins);
    
    // Saving the generated RandomForest model
    model.save(sc, args(2));
   }
}
