using Microsoft.ML;
using System.Collections.Generic;
using System;

namespace CustomTransform
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            var sampleData = new List<InputData>            
            {
                new InputData { Date = new DateTime(2019, 9, 19)},
                new InputData { Date = new DateTime(2019, 9, 20)},
                new InputData { Date = new DateTime(2019, 9, 21)},
                new InputData { Date = new DateTime(2019, 9, 22)}                                
            };

            var data = mlContext.Data.LoadFromEnumerable(sampleData);

            Action<InputData, MappingOutput> mapping = (input, output) =>
                output.IsWeekend = (input.Date.DayOfWeek == DayOfWeek.Saturday || input.Date.DayOfWeek == DayOfWeek.Sunday);

            var pipeline = mlContext.Transforms.CustomMapping(mapping, "customMappingIsWeekend");

            var transformData = pipeline.Fit(data).Transform(data);

            var enumData = mlContext.Data.CreateEnumerable<WeekendData>(
                                transformData, reuseRowObject: true);

            foreach (var row in enumData)
            {
                System.Console.WriteLine($"{row.Date} - {row.IsWeekend}");
            }
        }
    }
}