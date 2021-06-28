using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Image;

namespace TransferLearningTF
{
    class Program
    {
        static readonly string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
        static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        static readonly string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");
        static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        static readonly string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
        static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "whoisthis.jpg");
        // static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "thanos.jpg");
        static readonly string _modelFile = Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        static readonly string _inputImageClassifierZip = Path.Combine(_assetsPath, "inputs-predict", "imageClassifier.zip");
        static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);

            var model = ReuseAndTuneInceptionModel(mlContext, _trainTagsTsv, _trainImagesFolder, _modelFile, _outputImageClassifierZip);

            ClassifyImages(mlContext, _predictImageListTsv, _predictImagesFolder, _outputImageClassifierZip, model);

            ClassifySingleImage(mlContext, _predictSingleImage, _outputImageClassifierZip, model);
        }

        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }

        // Build and train model
        public static ITransformer ReuseAndTuneInceptionModel(MLContext mlContext, string trainDataFilePath, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {

            var trainData = mlContext.Data.LoadFromTextFile<ImageData>(path: trainDataFilePath, hasHeader: false);

            // First, the image transforms transform the images into the model's expected format.
            // The ScoreTensorFlowModel transform scores the TensorFlow model and allows communication 
            var imgProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
                    .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
                    .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                    .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                    .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation).
                                ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, 
                                                     inputColumnNames: new[] { "input" }, 
                                                     addBatchDimensionInput: true))
                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "softmax2_pre_activation"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
                    .AppendCacheCheckpoint(mlContext);

            Console.WriteLine("=============== Training classification model ===============");
            ITransformer trainedModel = imgProcessPipeline.Fit(trainData);

            var predictions = trainedModel.Transform(trainData);

            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
            DisplayResults(imagePredictionData);

            Console.WriteLine("=============== Classification metrics ===============");
            var multiclassContext = mlContext.MulticlassClassification;
            var metrics = multiclassContext.Evaluate(predictions, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");

            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

            return trainedModel;
        }

        public static void ClassifyImages(MLContext mlContext, string dataLocation, string imagesFolder, string outputModelLocation, ITransformer model)
        {

            // Read the image_list.tsv file and add the filepath to the image file name 
            // before loading into ImageData 
            var imageData = ReadFromTsv(dataLocation, imagesFolder);
            var imageDataView = mlContext.Data.LoadFromEnumerable<ImageData>(imageData);

            var predictions = model.Transform(imageDataView);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);

            Console.WriteLine("=============== Making classifications ===============");

            DisplayResults(imagePredictionData);
        }

        public static void ClassifySingleImage(MLContext mlContext, string imagePath, string outputModelLocation, ITransformer model)
        {
            // load the fully qualified image file name into ImageData 
            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };

            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);

            Console.WriteLine("=============== Making single image classification ===============");
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }

        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }

        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            //Need to parse through the tags.tsv file to combine the file path to the 
            // image name for the ImagePath property so that the image file can be found.
            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData()
             {
                 ImagePath = Path.Combine(folder, line[0])
             });
        }
    }
}