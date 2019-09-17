using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers;
using Common;

namespace EmployeeAttrition
{
    internal static class Program
    {

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            IDataView attritionData = mlContext.Data.LoadFromTextFile<Employee>(path: "./data/attrition.csv", hasHeader: true, separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(attritionData, testFraction: 0.2);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var numFields = attritionData.Schema.AsEnumerable()
                .Select(column => new { column.Name, column.Type })
                .Where(column => (column.Name != nameof(Employee.Attrition)) && (column.Type.ToString() == "Single"))
                .ToArray();

            var numFieldNames = numFields.AsEnumerable()
                .Select(column => column.Name)
                .ToList();

            var oheFieldNames = new List<string>();
            oheFieldNames.Add("OHE-" + nameof(Employee.BusinessTravel));
            oheFieldNames.Add("OHE-" + nameof(Employee.Department));
            oheFieldNames.Add("OHE-" + nameof(Employee.EducationField));
            oheFieldNames.Add("OHE-" + nameof(Employee.MaritalStatus));
            oheFieldNames.Add("OHE-" + nameof(Employee.JobLevel));
            oheFieldNames.Add("OHE-" + nameof(Employee.JobRole));
            oheFieldNames.Add("OHE-" + nameof(Employee.OverTime));

            var allFeatureFields = new List<string>();
            allFeatureFields.AddRange(oheFieldNames);
            string[] numFeatures = numFieldNames.ToArray();
            allFeatureFields.AddRange(numFeatures);
            string[] allFeatureNames = allFeatureFields.ToArray();

            IEstimator<ITransformer> featurizePipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair("OHE-" + nameof(Employee.BusinessTravel), nameof(Employee.BusinessTravel)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.Department), nameof(Employee.Department)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.EducationField), nameof(Employee.EducationField)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.MaritalStatus), nameof(Employee.MaritalStatus)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.JobLevel), nameof(Employee.JobLevel)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.JobRole), nameof(Employee.JobRole)),
                    new InputOutputColumnPair("OHE-" + nameof(Employee.OverTime), nameof(Employee.OverTime))
                }, OneHotEncodingEstimator.OutputKind.Indicator);

            featurizePipeline = featurizePipeline.Append(mlContext.Transforms.Concatenate("Features", allFeatureNames))
                                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                .AppendCacheCheckpoint(mlContext);

            ConsoleHelper.ConsoleWriteHeader("=============== Begin to train the model ===============");

            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                            labelColumnName: nameof(Employee.Attrition),
                            featureColumnName: "Features");

            /* ----- Tried with other trainers below and compared the outcome ------ */
            // var trainer = mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: nameof(Employee.Attrition), featureColumnName: "Features");
            // var trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(Employee.Attrition), featureColumnName: "Features");
            // var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: nameof(Employee.Attrition), featureColumnName: "Features");
            // var trainer = mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: nameof(Employee.Attrition), featureColumnName: "Features");
            /* ------------------------------------------------------------------- */

            var trainPipeline = featurizePipeline.Append(trainer);
            var trainedModel = trainPipeline.Fit(trainData);

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

            var testDataPredictions = trainedModel.Transform(testData);
            var evaluateMetrics = mlContext.BinaryClassification.Evaluate(data: testDataPredictions,
                                                                labelColumnName: nameof(Employee.Attrition),
                                                                scoreColumnName: "Score");
            ConsoleHelper.PrintBinaryClassificationMetrics(trainedModel.ToString(), evaluateMetrics);

            Console.WriteLine("===== Permutation Test =====");

            var transformedData = trainedModel.Transform(trainData);
            var permutationMetrics = mlContext.BinaryClassification.PermutationFeatureImportance(
                    predictionTransformer: trainedModel.LastTransformer,
                    data: transformedData,
                    labelColumnName: nameof(Employee.Attrition),
                    permutationCount: 50);

            var mapFields = new List<string>();
            for (int i = 0; i < allFeatureNames.Count(); i++)
            {
                var slotField = new VBuffer<ReadOnlyMemory<char>>();
                if (transformedData.Schema[allFeatureNames[i]].HasSlotNames())
                {
                    transformedData.Schema[allFeatureNames[i]].GetSlotNames(ref slotField);
                    for (int j = 0; j < slotField.Length; j++)
                    {
                        mapFields.Add(allFeatureNames[i]);
                    }
                }
                else
                {
                    mapFields.Add(allFeatureNames[i]);
                }
            }

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                .OrderByDescending(
                feature => Math.Abs(feature.AreaUnderRocCurve.Mean));

            foreach (var feature in sortedIndices)
            {
                Console.WriteLine($"{mapFields[feature.index],-20}|\t{Math.Abs(feature.AreaUnderRocCurve.Mean):F6}");
            }
        }
    }
}
