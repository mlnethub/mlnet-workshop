using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MovieRecommender
{
    public static class HyperParameterExplorer
    {
        public static void HyperParameterExploration(
            MLContext mlContext,
            IEstimator<ITransformer> dataProcessingPipeline,
            IDataView trainingDataView)
        {
            var results = new List<(double rootMeanSquaredError,
                                    double rSquared,
                                    int iterations,
                                    int approximationRank)>();

            for (int iterations = 5; iterations < 100; iterations += 5)
            {
                Console.WriteLine($"Iterations: {iterations}");
                for (int approximationRank = 50; approximationRank < 250; approximationRank += 50)
                {
                    var option = new MatrixFactorizationTrainer.Options
                    {
                        MatrixColumnIndexColumnName = "userIdEncoded",
                        MatrixRowIndexColumnName = "movieIdEncoded",
                        LabelColumnName = "Label",
                        NumberOfIterations = iterations,
                        ApproximationRank = approximationRank,
                        Quiet = true
                    };

                    var currentTrainer = mlContext.Recommendation().Trainers.MatrixFactorization(option);

                    var completePipeline = dataProcessingPipeline.Append(currentTrainer);

                    var crossValMetrics = mlContext.Recommendation().CrossValidate(
                                            trainingDataView,
                                            completePipeline,
                                            labelColumnName: "Label");

                    results.Add(
                                (crossValMetrics.Average(m => m.Metrics.RootMeanSquaredError),
                                 crossValMetrics.Average(m => m.Metrics.RSquared),
                                 iterations,
                                 approximationRank
                                )
                               );
                }
            }

            Console.WriteLine("--- Hyper parameters and metrics ---");

            foreach (var result in results.OrderByDescending(r => r.rSquared))
            {
                Console.Write($"NumberOfIterations: {result.iterations}");
                Console.Write($" ApproximationRank: {result.approximationRank}");
                Console.Write($" RootMeanSquaredError: {result.rootMeanSquaredError}");
                Console.WriteLine($" RSquared: {result.rSquared}");
            }

            Console.WriteLine();
            Console.WriteLine("Done");
        }
    }    
}