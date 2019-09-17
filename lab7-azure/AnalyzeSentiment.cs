using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Microsoft.Extensions.ML;
using SentimentAnalysisApp.DataModels;

namespace SentimentAnalysisApp
{
    public class AnalyzeSentiment
    {
        private readonly PredictionEnginePool<SentimentIssue, SentimentPrediction> _predictionEnginePool;

        // AnalyzeSentiment class constructor
        public AnalyzeSentiment(PredictionEnginePool<SentimentIssue, SentimentPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [FunctionName("AnalyzeSentiment")]
        public async Task<IActionResult> Run(
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = null)] HttpRequest req,
        ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            //Parse HTTP Request Body
            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            SentimentIssue data = JsonConvert.DeserializeObject<SentimentIssue>(requestBody);
            
            
            //Make Prediction
            SentimentPrediction prediction = _predictionEnginePool.Predict(data);

            //Convert prediction to string
            string sentiment = Convert.ToBoolean(prediction.Prediction) ? "Toxic" : "Non-Toxic";

            //Return Prediction
            return (ActionResult)new OkObjectResult(sentiment);
        }
    }
}
