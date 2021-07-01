//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace SampleClassification.Model
{
    public class ModelInput
    {
        [ColumnName("sentiment"), LoadColumn(0)]
        public string Sentiment { get; set; }


        [ColumnName("sentiment_label"), LoadColumn(1)]
        public string Sentiment_label { get; set; }


    }
}
