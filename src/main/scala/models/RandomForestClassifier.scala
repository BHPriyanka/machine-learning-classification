package models

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object RandomForestClassifier {

	def main(args : Array[String]) {
		val spark = SparkSession.builder()
				.appName("Neighbourhood Vectors application")
				.master("local[*]")
				.config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
				.getOrCreate()

				System.setProperty("hadoop.home.dir", "C:/winutils");

		val sc = spark.sparkContext

		 // Load and parse the data file.
    val trainingData = MLUtils.loadLibSVMFile(sc, "data/image1/libSVM/part-00000")
    // Split the data into training and test sets (30% held out for testing)
    //val splits = data.randomSplit(Array(0.7, 0.3))
    //val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    /*val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)*/
    println("Learned classification forest model:\n" + model.toDebugString)

    // Save and load model
    model.save(sc, "target/tmp/myRandomForestClassificationModel")
  
    /*val metrics = new MulticlassMetrics(labelAndPreds)
    val accuracy = metrics.accuracy
    println("Accuracy: " +accuracy)*/
    
    //testData.saveAsTextFile("data/image1/randomforest/testData")
    //labelAndPreds.saveAsTextFile("data/image1/labelAndPreds")
    sc.stop()
	}
}