//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using System;
using SampleRegression.Model;

namespace SampleRegression.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create single instance of sample data from first line of dataset for model input
            ModelInput sampleData = new ModelInput()
            {
                Season = 1F,
                Yr = 0F,
                Mnth = 1F,
                Hr = 0F,
                Holiday = 0F,
                Weekday = 6F,
                Workingday = 0F,
                Weathersit = 1F,
                Temp = 0.24F,
                Atemp = 0.2879F,
                Hum = 0.81F,
                Windspeed = 0F,
            };

            // Make a single prediction on the sample data and print results
            var predictionResult = ConsumeModel.Predict(sampleData);

            Console.WriteLine("Using model to make single prediction -- Comparing actual Cnt with predicted Cnt from sample data...\n\n");
            Console.WriteLine($"Season: {sampleData.Season}");
            Console.WriteLine($"Yr: {sampleData.Yr}");
            Console.WriteLine($"Mnth: {sampleData.Mnth}");
            Console.WriteLine($"Hr: {sampleData.Hr}");
            Console.WriteLine($"Holiday: {sampleData.Holiday}");
            Console.WriteLine($"Weekday: {sampleData.Weekday}");
            Console.WriteLine($"Workingday: {sampleData.Workingday}");
            Console.WriteLine($"Weathersit: {sampleData.Weathersit}");
            Console.WriteLine($"Temp: {sampleData.Temp}");
            Console.WriteLine($"Atemp: {sampleData.Atemp}");
            Console.WriteLine($"Hum: {sampleData.Hum}");
            Console.WriteLine($"Windspeed: {sampleData.Windspeed}");
            Console.WriteLine($"\n\nPredicted Cnt: {predictionResult.Score}\n\n");
            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
