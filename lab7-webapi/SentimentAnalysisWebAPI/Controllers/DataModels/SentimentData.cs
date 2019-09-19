using Microsoft.ML.Data;

namespace SentimentAnalysisWebAPI.DataModels
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string SentimentText;

        [LoadColumn(1)]
        [ColumnName("Label")]
        public bool Sentiment;
    }
}