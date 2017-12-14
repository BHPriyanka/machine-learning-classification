package models

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object DecisionTreeClassifier {
  
    def main(args : Array[String]) {
	    val spark = SparkSession.builder()
			    .appName("Neighbourhood Vectors application")
  			  .master("local[*]")
			     .config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
			    .getOrCreate()

		  System.setProperty("hadoop.home.dir", "C:/winutils");

	  	val sc = spark.sparkContext
	  	
	  	val startTime = System.currentTimeMillis();
	  	
	  	// Load and parse the data
      val trainingData = MLUtils.loadLibSVMFile(sc, args(0))
      val testData = MLUtils.loadLibSVMFile(sc, args(1));
      // Split the data into training and test sets (30% held out for testing)
      //val splits = data.randomSplit(Array(0.7, 0.3))
      //val (trainingData, testData) = (splits(0), splits(1))

      // Train a DecisionTree model.
      //  Empty categoricalFeaturesInfo indicates all features are continuous.
      val numClasses = 2
      val categoricalFeaturesInfo = Map[Int, Int]()
      val impurity = "gini"
      val maxDepth = 12
      val maxBins = 32

      val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
        impurity, maxDepth, maxBins)

      // Evaluate model on test instances and compute test error
      val labelAndPreds = testData.map { point =>
        val prediction = model.predict(point.features)
        (point.label, prediction)
      }
      val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
      val endTime = System.currentTimeMillis();
      println("Test Error = " + testErr)
      //println("Learned classification tree model:\n" + model.toDebugString)

      // Save and load model
      //model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
      
      val metrics = new MulticlassMetrics(labelAndPreds)
      val accuracy = metrics.accuracy
      println("Accuracy: " +accuracy)
      println("Running time: "+ (endTime-startTime) + " maxDepth : " + maxDepth);
      println("Default number of partitions for training Data: " +trainingData.getNumPartitions)
      println("Default number of partitions for test Data: " + testData.getNumPartitions)
      
      //testData.saveAsTextFile("data/image1/decisiontree/testData")
      //labelAndPreds.saveAsTextFile("data/image1/decisiontree/labelAndPreds")
      sc.stop() 
      }
}