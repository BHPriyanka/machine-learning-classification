package models

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
// $example off$
import org.apache.spark.sql.SparkSession

object NaiveBayesExample {

  def main(args: Array[String]): Unit = {
    //val conf = new SparkConf().setAppName("NaiveBayesExample").setMaster("local");
    val spark = SparkSession.builder()
		    .appName("Naive Bayes")
  		  .master("local[*]")
		     .config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
		    .getOrCreate()
       
    //val sc = new SparkContext(conf)
		val sc = spark.sparkContext
    System.setProperty("hadoop.home.dir", "C:/winutils");
    
    
    val startTime = System.currentTimeMillis();

    // Load and parse the data file.
    val trainingData = MLUtils.loadLibSVMFile(sc, args(0))
    val testData = MLUtils.loadLibSVMFile(sc, args(1));
    
    // Split data into training (60%) and test (40%).
    //val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

    val model = NaiveBayes.train(trainingData, lambda = 1.0, modelType = "multinomial")

    val predictionAndLabel = testData.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / testData.count()
    val endTime = System.currentTimeMillis();
    println("ACCURACY" + accuracy);
    println("Running time: "+ (endTime-startTime) );
     
    // Save and load model
    //model.save(sc, "target/tmp/myNaiveBayesModel")
  
    sc.stop()
  }
}
