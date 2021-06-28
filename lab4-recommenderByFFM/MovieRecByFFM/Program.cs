using System;
using System.Drawing;
using MovieRecByFFM.DataStructures;
using Microsoft.ML;
using Common;

namespace MovieRecByFFM
{
    class Program
    {
        private static string dataFile = @"../Data/ratings.csv";
        private const string FEATURES = "Features";
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            //STEP 1-1: Read data, clearly separate the groups avoiding label leakage
            var sourceData = mlContext.Data.LoadFromTextFile<MovieRating>(path:dataFile, hasHeader:true, separatorChar: ',');
            var split = mlContext.Data.TrainTestSplit(sourceData, 0.2, samplingKeyColumnName:nameof(MovieRating.userId));
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            // ConsoleHelper.ShowDataViewInConsole(mlContext, trainData, 10);
            // ConsoleHelper.ShowDataViewInConsole(mlContext, testData, 10);

            // STEP 1-2: Custom Mapping to label Recommend or Not
            Action<MovieRating, MappingOutput> mapping = (input, output) =>
                output.Recommend = (input.rating > 3) ? true : false;

            var setRating = mlContext.Transforms.CustomMapping(mapping, "customRating");

            // To validate the custom mapping implementation
            // var transformData = setRating.Fit(trainData).Transform(trainData);
            // ConsoleHelper.ShowDataViewInConsole(mlContext, transformData, 10);

            // STEP 2: Establish training pipeline
            var featurize = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "UserIDFeaturized", inputColumnName:nameof(MovieRating.userId))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "MovieIDFeaturized", inputColumnName:nameof(MovieRating.movieId)))
                            .Append(mlContext.Transforms.Concatenate(FEATURES, "UserIDFeaturized", "MovieIDFeaturized"))
                            .Append(setRating);

            var trainer = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(
                            featureColumnNames: new string[] { FEATURES },
                            labelColumnName: nameof(MappingOutput.Recommend));
            var pipeline = featurize.Append(trainer);

            // STEP 3: Train
            var model = pipeline.Fit(trainData);

            // STEP 4: Evaluate
            var testResult = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(
                                data: testResult, 
                                labelColumnName: nameof(MappingOutput.Recommend), 
                                scoreColumnName: "Score", 
                                predictedLabelColumnName: "PredictedLabel");
                                
            Console.WriteLine("Evaluation Metrics: Accuracy:" + Math.Round(metrics.Accuracy, 2) + " AUC:" + Math.Round(metrics.AreaUnderRocCurve, 2));

            // STEP 5: Predict
            var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            MovieRating predictData = new MovieRating() { userId = "6", movieId = "1" };

            var movieRatingPrediction = predictionEngine.Predict(predictData);
            Console.WriteLine($"UserId:{predictData.userId} with movieId: {predictData.movieId} Score:{Sigmoid(movieRatingPrediction.Score)} and Recommended as {movieRatingPrediction.PredictedLabel}", Color.YellowGreen);
        }

        public static float Sigmoid(float x)
        {
            return (float)(100 / (1 + Math.Exp(-x)));
        }
    }
}
