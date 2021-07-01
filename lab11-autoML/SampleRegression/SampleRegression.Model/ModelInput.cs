//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace SampleRegression.Model
{
    public class ModelInput
    {
        [ColumnName("instant"), LoadColumn(0)]
        public float Instant { get; set; }


        [ColumnName("dteday"), LoadColumn(1)]
        public string Dteday { get; set; }


        [ColumnName("season"), LoadColumn(2)]
        public float Season { get; set; }


        [ColumnName("yr"), LoadColumn(3)]
        public float Yr { get; set; }


        [ColumnName("mnth"), LoadColumn(4)]
        public float Mnth { get; set; }


        [ColumnName("hr"), LoadColumn(5)]
        public float Hr { get; set; }


        [ColumnName("holiday"), LoadColumn(6)]
        public float Holiday { get; set; }


        [ColumnName("weekday"), LoadColumn(7)]
        public float Weekday { get; set; }


        [ColumnName("workingday"), LoadColumn(8)]
        public float Workingday { get; set; }


        [ColumnName("weathersit"), LoadColumn(9)]
        public float Weathersit { get; set; }


        [ColumnName("temp"), LoadColumn(10)]
        public float Temp { get; set; }


        [ColumnName("atemp"), LoadColumn(11)]
        public float Atemp { get; set; }


        [ColumnName("hum"), LoadColumn(12)]
        public float Hum { get; set; }


        [ColumnName("windspeed"), LoadColumn(13)]
        public float Windspeed { get; set; }


        [ColumnName("casual"), LoadColumn(14)]
        public float Casual { get; set; }


        [ColumnName("registered"), LoadColumn(15)]
        public float Registered { get; set; }


        [ColumnName("cnt"), LoadColumn(16)]
        public float Cnt { get; set; }


    }
}
