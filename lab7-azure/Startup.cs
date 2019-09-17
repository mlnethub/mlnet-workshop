using Microsoft.Azure.Functions.Extensions.DependencyInjection;
using Microsoft.Extensions.ML;
using SentimentAnalysisApp;
using SentimentAnalysisApp.DataModels;

[assembly: FunctionsStartup(typeof(Startup))]
namespace SentimentAnalysisApp
{
    public class Startup : FunctionsStartup
    {
        public override void Configure(IFunctionsHostBuilder builder)
        {
            builder.Services.AddPredictionEnginePool<SentimentIssue, SentimentPrediction>()
                .FromUri("https://rpmovietoxictest.blob.core.windows.net/models/sentiment_model.zip");
        }
    }
}