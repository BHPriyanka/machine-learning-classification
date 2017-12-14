package models

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils

/*
 * Author 1: Vishal Sanjiv Kotak
 * Author 2: Priyanka Bhadravathi Halesh
 * The below Scala program reads the generated Random Forest Training Model and
 * the test data. The locations of the mentioned files have been provided as
 * arguments. Every records from the test data set is passed to the model
 * and the predicted label is stored in an RDD. After completion of all the 
 * test records, the RDD is saved at the location provided as the argument.
 */

object RandomForestPrediction {
  def main(args: Array[String]): Unit = {
    
    // Creating the Spark Configuration object
     val conf = new SparkConf()
                    .setAppName("RandomForestClassificationExample")
                    .setMaster(args(0));
    
    // Creating the Spark Context object
    val sc = new SparkContext(conf);
    
    // Reading the test data in the LibSVM format (DecisionVector, LabelPoint)
    // where Decision Vector is the feature vector and label point is the
    // true label.
    val testData = MLUtils.loadLibSVMFile(sc, args(1));
     
    // Loading the Random Forest model from the argument
     val randomForestTrainedModel = RandomForestModel.load(sc, args(2));
     
     // Applying the saved model on each record of the test data.
     // This prediction is stored in the RDD with the true label
     // point for  validation data and ? for test data
     val predictedLabelsAndTrueLabels = testData.map { point =>
        val prediction = randomForestTrainedModel.predict(point.features);
        (point.label, prediction);
     }
     
     // Finding out the accuracy of the validation data by comparing the true
     // label and prediction and dividing them by the count of the total
     // test data.
     val accuracy = predictedLabelsAndTrueLabels.filter(r => r._1 == r._2)
                                                .count.toDouble / testData.count();
     
     // Printing the accuracy
     println("Accuracy " + accuracy);
     
     // Removing the prediction from the RDD and converting it to String RDD
     // to save it the required format
     predictedLabelsAndTrueLabels
           .map{ point => point._2.toInt.toString()}  
           .saveAsTextFile(args(3));
     
  }
}
