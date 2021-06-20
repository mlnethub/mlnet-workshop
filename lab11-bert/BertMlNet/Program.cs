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

            var model = new Bert(vocabularyFilePath, modelFilePath);

            var (tokens, probability) = model.Predict(args[0], args[1]);

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