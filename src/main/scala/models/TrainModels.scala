package models

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, DoubleType};
import org.apache.spark.sql.Row
import org.apache.spark.sql.Row

object TrainModels {

	def main(args : Array[String]) {
		val spark = SparkSession.builder()
				.appName("Neighbourhood Vectors application")
				.master("local[*]")
				.config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
				.getOrCreate()

				System.setProperty("hadoop.home.dir", "C:/winutils");

		val sc = spark.sparkContext
		
		val testData = MLUtils.loadLibSVMFile(sc, "data/image6/libSVM/part-00000")
		
		// Load Random Forest model
	  val randomForestModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
	  
	  // Evaluate model on test instances and compute test error
    val RFLabelAndPreds = testData.map { point =>
      val prediction = randomForestModel.predict(point.features)
      (point.label, prediction)
    }
    val RFTestErr = RFLabelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + RFTestErr)
    
    val RFMetrics = new MulticlassMetrics(RFLabelAndPreds)
    val RFAccuracy = RFMetrics.accuracy
    println("Accuracy: " +RFAccuracy)
	  
	  // Load Decision Tree Model
	  val decisionTreeModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
	  
    // Evaluate model on test instances and compute test error
    val DTLabelAndPreds = testData.map { point =>
       val prediction = decisionTreeModel.predict(point.features)
       (point.label, prediction)
    }
    val DTTestErr = DTLabelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println("Test Error = " + DTTestErr)
    
    val DTMetrics = new MulticlassMetrics(DTLabelAndPreds)
    val DTAccuracy = DTMetrics.accuracy
    println("Accuracy: " +DTAccuracy)
	  
 	  //Load Naive-Bayes model
	  //val naiveBayesModel = NaiveBayesModel.load(sc, "target/tmp/myNaiveBayesModel")

    // Join RDDs and create a DF with schema
    val RFLabelPredPair = RFLabelAndPreds.map(entry => (entry._1, entry._2))
    val DTLabelPredPair = DTLabelAndPreds.map(entry => (entry._1, entry._2))
    val predictions = RFLabelPredPair.join(DTLabelPredPair)
    println(RFLabelPredPair.count())
    println(DTLabelPredPair.count())
         
    RFLabelPredPair.saveAsTextFile("data/RFLabelPredPair")
    DTLabelPredPair.saveAsTextFile("data/DTLabelPredPair")
        
        
	}
}