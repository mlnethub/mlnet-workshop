using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using SentimentWebAPI.DataModels;

namespace SentimentWebAPI.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class PredictController : ControllerBase
    {
        private readonly PredictionEnginePool<SentimentData, SentimentPrediction> _predictionEnginePool;

        public PredictController(PredictionEnginePool<SentimentData,SentimentPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        // GET Predictor/sentimentprediction?sentimentText=ML.NET is awesome!
        [HttpGet]
        [Route("sentimentprediction")]
        public ActionResult<string> PredictSentiment([FromQuery]string sentimentText)
        {
            SentimentData sampleData = new SentimentData() { SentimentText = sentimentText };

            //Predict sentiment
            SentimentPrediction prediction = _predictionEnginePool.Predict(sampleData);

            bool isPositive = prediction.Sentiment;
            float probability = CalculatePercentage(prediction.Score);
            string retVal = $"Prediction: Postive?: '{isPositive.ToString()}' with {probability.ToString()}% probability of toxicity for the text '{sentimentText}'";

            return retVal;

        }

        public static float CalculatePercentage(double value)
        {
            return 100 * (1.0f / (1.0f + (float)Math.Exp(-value)));
        }
    }    
}