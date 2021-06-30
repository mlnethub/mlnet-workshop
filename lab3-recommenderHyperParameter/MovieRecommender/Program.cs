using System;
using Microsoft.ML;
using MovieRecommender.DataStructures;
using System.IO;
using Microsoft.ML.Trainers;
using static MovieRecommender.HyperParameterExplorer;

namespace MovieRecommender
{
    class Program
    {
        // Using the ml-latest-small.zip as dataset from https://grouplens.org/datasets/movielens/. 
        private static string ModelsRelativePath = @"../../../../MLModels";
        public static string DatasetsRelativePath = @"../../../../Data";

        private static string TrainingDataRelativePath = $"{DatasetsRelativePath}/recommendation-ratings-train.csv";
        private static string TestDataRelativePath = $"{DatasetsRelativePath}/recommendation-ratings-test.csv";
        private static string MoviesDataLocation = $"{DatasetsRelativePath}/movies.csv";

        private static string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
        private static string TestDataLocation = GetAbsolutePath(TestDataRelativePath);

        private static string ModelPath = GetAbsolutePath(ModelsRelativePath);

        private const float predictionuserId = 6;
        private const int predictionmovieId = 10;

        static void Main(string[] args)
        {
            MLContext mlcontext = new MLContext();

            IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(
                                            TrainingDataLocation, 
                                            hasHeader: true, 
                                            separatorChar:',');

            var dataProcessingPipeline = mlcontext.Transforms.Conversion.MapValueToKey(
                                            outputColumnName: "userIdEncoded", 
                                            inputColumnName: nameof(MovieRating.userId))
                                        .Append(mlcontext.Transforms.Conversion.MapValueToKey(
                                            outputColumnName: "movieIdEncoded", 
                                            inputColumnName: nameof(MovieRating.movieId)));
            
            HyperParameterExploration(mlcontext, dataProcessingPipeline, trainingDataView);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
