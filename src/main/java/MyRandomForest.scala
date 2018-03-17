import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object MyRandomForest {

  def main(args: Array[String]): Unit = {


    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val inputFile = "data/train_users_2.csv";

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    import sparkSession.implicits._

    //Define the scheme of the data
    var airUsersSchema = StructType(Array(
      StructField("id", StringType, true),
      StructField("date_account_created", StringType, true),
      StructField("timestamp_first_active", StringType, true),
      StructField("date_first_booking", StringType, true),
      StructField("gender", StringType, true),
      StructField("age", StringType, true),
      StructField("signup_method", StringType, true),
      StructField("signup_flow", StringType, true),
      StructField("affilate_provider", StringType, true),
      StructField("affilate_channel", StringType, true),
      StructField("first_affilate_tracked", StringType, true),
      StructField("signup_app", StringType, true),
      StructField("signup_ap1p", StringType, true),
      StructField("first_device_type", StringType, true),
      StructField("first_browser", StringType, true),
      StructField("country_destination", StringType, true)
    ));

    //Read CSV file to DF and define scheme on the fly
    val categories = Seq("US", "FR", "CA", "GB", "ES", "IT", "PT", "NL", "DE", "AU", "NDF")

    val train_users = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .schema(airUsersSchema)
      .csv(inputFile)
      .filter(x => (!x.anyNull))
      .filter("gender <> '-unknown-'")
      .filter("gender <> 'OTHER'")
      .withColumn("intAge", 'age.cast("Int")) // exclude double ages
      .filter(col("country_destination").isin(categories: _*))
      .filter($"intAge" < 80)
      .withColumn("quarter", quarter(col("date_first_booking")))

    val genderIndexer = new StringIndexer()
      .setInputCol("gender")
      .setOutputCol("igender")

    val affiliateIndexer = new StringIndexer()
      .setInputCol("affilate_channel")
      .setOutputCol("iaffilate_channel")

    val countryIndexer = new StringIndexer()
      .setInputCol("country_destination")
      .setOutputCol("icountry_destination")

    val assembler = new VectorAssembler()
      .setInputCols(Array("igender", "intAge", "iaffilate_channel", "quarter"))
      .setOutputCol("features")
    val Array(trainingData, testData) = train_users.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("icountry_destination")
      .setFeaturesCol("features")
      .setNumTrees(10)

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(genderIndexer, affiliateIndexer, countryIndexer, assembler, rf))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("icountry_destination")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    //Train model. This also runs the indexer.
    val model = cv.fit(trainingData)

    //Make predictions.
    val predictions = model.transform(testData)
    predictions.show()

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy)
    println("Test Error = " + (1.0 - accuracy))
  }
}
