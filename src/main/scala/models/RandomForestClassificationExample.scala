package models

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object RandomForestClassificationExample {
  def main(args: Array[String]): Unit = {
    
    //val conf = new SparkConf().setAppName("RandomForestClassificationExample").setMaster("local[*]");
    val spark = SparkSession.builder()
			    .appName("RandomForestClassificationExample")
  			  .master("local[*]")
			    .config("spark.sql.warehouse.dir", "file:///C:/Users/priya/workspace/Assignment5/park-warehouse")
			    .getOrCreate()

    
    //val sc = new SparkContext(conf);
	  val sc = spark.sparkContext
    
    System.setProperty("hadoop.home.dir", "C:/winutils");
    

    
    val trainingData = MLUtils.loadLibSVMFile(sc, args(0), -1, 15);
       
    val testData = MLUtils.loadLibSVMFile(sc, args(1), -1, 5);
     
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]();
    // Number of trees in the forest.(default 3)
    val numTrees = 20;
    val featureSubsetStrategy = "auto";
    val impurity = "gini";
    //default 4
    val maxDepth = 4
    val maxBins = 32
    val startTime = System.currentTimeMillis();
    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
                                  numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins);
     val endTime = System.currentTimeMillis();
    // Save and load model
    model.save(sc, "./target/tmp/myRandomForestClassificationModel")
    
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    
    val accuracy = labelAndPreds.filter(r => r._1 == r._2).count.toDouble / testData.count();
   
        
    println("Accuracy " + accuracy); // 0.995945483
    println("Running time: "+ (endTime-startTime) + " numTrees: " + numTrees + " maxDepth : " + maxDepth);
    println("Default number of partitions for training Data: " +trainingData.getNumPartitions)
    println("Default number of partitions for test Data: " + testData.getNumPartitions)
    sc.stop()
  }
}
// Save and load model
    // 
    // val sameModel = RandomForestModel.load(sc, "target/tmp/myRandomForestClassificationModel")
    // $example off$

// model.save(sc, args(1));
//    val numTrees1 = 12;
//    val maxDepth1 = 8;
//    
//    val model1 = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
//                                  numTrees1, featureSubsetStrategy, impurity, maxDepth1, maxBins);
//    
//    val labelAndPreds1 = testData.map { point =>
//      val prediction = model1.predict(point.features)
//      (point.label, prediction)
//    }
//    
//    val accuracy1 = labelAndPreds1.filter(r => r._1 == r._2).count.toDouble / testData.count();
//    println("Accuracy1 " + accuracy1)//0.9964339797762689
