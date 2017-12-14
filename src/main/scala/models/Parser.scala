package models

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.util.MLUtils

/*
 * Author 1: Priyanka Bhadravathi Halesh 
 * Author 2: Vishal Sanjiv Kotak
 * The below code reads the data from the .csv file and converts it
 * to the LibSVM format required to train the model in the Spark MLIB
 * packages. This can be considered as a pre-processing step and is 
 * primarily done to reduce the execution time during model tuning.
 */

object Parser {
  
   def main(args: Array[String]): Unit = {
  
   // Creating the Spark Configuration Object
   val spark = SparkSession.builder()
			    .appName("Parser")
  			    .master(args(0))
			    .getOrCreate();
   
		// Creating the Spark Context objext
	  val sc = spark.sparkContext;
	  
	  // Creating a data frame by reading the csv file and
	  // providing the parameters to disable the header 
	  // providing the csv format type.
	  val df = spark.sqlContext
	                .read
	                .format("com.databricks.spark.csv")
	                .option("header", "false")
	                .load(args(1));
   
	 // Converting the Data Frame to RDD
    val c = df.rdd
    
    // Converting the RDD to Lib SVM format where the LabelPoint is the data column
    // at 3088 position and the feature set is a dense vector from 0 to 3087 of the 
    // data column
    val training = c.map(row => (LabeledPoint((row.get(3087)).toString().toDouble, 
        Vectors.dense(getBrightness(row)))));
    
    // Save the generated Lib SVM data set in the desired file location
    MLUtils.saveAsLibSVMFile(training, args(2));
   }
   
   // getBrightness function converts the csv file column to an array so that 
   // it can be converted into dense vector
   def getBrightness(row: Row): Array[Double] = {
     
     // Creating an array
      var b = new ArrayBuffer[Double]()
      
      // Converting individual data column string value to 
      // Double as dense vectors accept only double values
      for(i <- 0 to 3086) {
          b += (row.get(i)).toString().toDouble
      }
      
      // Return the array
      return b.toArray
      
    }
  
}
