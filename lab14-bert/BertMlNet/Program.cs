using System;
using System.IO;
using System.Text.Json;

namespace BertMlNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var assetsRelativePath = @"../../../Assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);
            var modelFilePath = Path.Combine(assetsPath, "Model", "bertsquad-10.onnx");
            var vocabularyFilePath = Path.Combine(assetsPath, "Vocabulary", "vocab.txt");
            var questionFilePath = Path.Combine(assetsPath, "Context", "question.txt");
            var dataFilePath = Path.Combine(assetsPath, "Context", "context.txt");

            var model = new Bert(vocabularyFilePath, modelFilePath);

            var contextText = "";
            var questionText = "";

            if (args == null || args.Length == 0)
            {
                contextText = File.ReadAllText(dataFilePath);
                questionText = File.ReadAllText(questionFilePath);
            }
            else 
            {
                contextText = args[0];
                questionText = args[1];
            }

            var (tokens, probability) = model.Predict(context:contextText, question:questionText);

            Console.WriteLine(JsonSerializer.Serialize(new
            {
                Probability = probability,
                Tokens = tokens
            }));
        }


        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

    }
}