using Microsoft.ML;
using System;
using System.Linq;

namespace ModelExplainability
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader: true, separatorChar: ',');

            var featureColumns = data.Schema
                .Select(col => col.Name)
                .Where(colName => colName != "Label" && colName != "OceanProximity")
                .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression());

            var model = pipeline.Fit(data);
            var transformedData = model.Transform(data);

            // Get weights of model
            var linearModel = model.LastTransformer.Model;

            var weights = linearModel.Weights;

            Console.WriteLine("Weights:");
            // Order features by importance
            var featureWeights = weights
                                .Select((item, index) => new { index, item })
                                .OrderByDescending(theWeight => Math.Abs(theWeight.item));

            foreach (var featureWeight in featureWeights)
            {
                Console.WriteLine($"Feature {featureColumns[featureWeight.index]} has weight {featureWeight.item}");
            }
            Console.WriteLine(Environment.NewLine);

            // Get global feature importance
            var lastTransformer = model.LastTransformer;
            var featureImportance = context.Regression.PermutationFeatureImportance(lastTransformer, transformedData);

            // Order features by importance
            var featureImportanceMetrics =
                featureImportance
                    .Select((metric, index) => new { index, metric.RSquared })
                    .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean));

            Console.WriteLine("Feature\tPFI");
            foreach (var feature in featureImportanceMetrics)
            {
                Console.WriteLine($"{featureColumns[feature.index],-20}|\t{feature.RSquared.Mean:F6}");
            }
            Console.WriteLine(Environment.NewLine);

            // Get feature importance for each row
            var firstRow = model.Transform(context.Data.TakeRows(transformedData, 1));

            var featureContribution = context.Transforms.CalculateFeatureContribution(lastTransformer, normalize: false);

            var featureContributionTransformer = featureContribution.Fit(firstRow);

            var featureContributionPipeline = model.Append(featureContributionTransformer);

            var predictionEngine = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(featureContributionPipeline);

            var sampleData = new HousingData
            {
                Longitude = -122.25f,
                Latitude = 37.85f,
                HousingMedianAge = 55.0f,
                TotalRooms = 1627.0f,
                TotalBedrooms = 235.0f,
                Population = 322.0f,
                Households = 120.0f,
                MedianIncome = 8.3014f,
                OceanProximity = "NEAR BAY"
            };

            var prediction = predictionEngine.Predict(sampleData);

            Console.WriteLine("Row feature importance:");

            // Order features by importance
            var predictContributions = prediction.FeatureContributions
                                .Select((item, index) => new { index, item })
                                .OrderByDescending(theWeight => Math.Abs(theWeight.item));

            foreach (var predictContribution in predictContributions)
            {
                Console.WriteLine($"Feature {featureColumns[predictContribution.index]} has feature contribution of {predictContribution.item}");
            }
        }
    }
}
