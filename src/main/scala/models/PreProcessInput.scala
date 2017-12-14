package models

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession
import scala.Double
import org.apache.spark.sql._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}

object SVM {
  
    def main(args : Array[String]) {
	    val spark = SparkSession.builder()
			    .appName("Classification application")
  			  .master("local[*]")
			    .config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
			    .getOrCreate()

		  System.setProperty("hadoop.home.dir", "C:/winutils");

		val sc = spark.sparkContext
				
		val df = spark.sqlContext.read.format("com.databricks.spark.csv").option("header", "false").load("./data/image6/L6_6_972760.csv").limit(2000)

		//FROM DATAFRAME TO RDD
    val c = df.rdd
    val image = c.map(row => (LabeledPoint((row.get(3087)).toString().toDouble, Vectors.dense(getBrightness(row)))))// this command will convert your dataframe in a RDD
    
    MLUtils.saveAsLibSVMFile(image, "./data/image6/libSVM/")
   }
    
  
    def getBrightness(row: Row): Array[Double] = {
      var b = new ArrayBuffer[Double]()
      for(i <- 0 to 3086) {
          b += (row.get(i)).toString().toDouble
      }
      
      return b.toArray
      
    }
}