using System;
using System.IO;
using Microsoft.ML;
using SampleBinaryClassification.Model.DataModels;

namespace classfierTest
{
    public class Program
    {
        private static readonly string ModelRelativePath = @"../../../../SampleBinaryClassification/SampleBinaryClassification.Model/MLModel.zip";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            DataViewSchema modelSchema;
            var sentimentModel = mlContext.Model.Load(ModelPath, out modelSchema);

            // ModelInput sampleStatement = new ModelInput { Sentiment = "You crappy *0(*&^&%^&%" };
            ModelInput sampleStatement = new ModelInput { Sentiment = "Not the best, imo" };

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(sentimentModel);

            // Score
            var resultprediction = predEngine.Predict(sampleStatement);

            Console.WriteLine($"=============== Single Prediction  ===============");
            Console.WriteLine($"Text: {sampleStatement.Sentiment} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Score of being toxic: {resultprediction.Score} ");
            Console.WriteLine($"================End of Process.Hit any key to exit==================================");
            Console.ReadLine();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath , relativePath);

            return fullPath;
        }
    }
}
