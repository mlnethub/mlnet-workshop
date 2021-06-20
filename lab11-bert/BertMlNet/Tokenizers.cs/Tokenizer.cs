using BertMlNet.Extensions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BertMlNet.Tokenizers
{
    public class Tokenizer
    {
        private readonly List<string> _vocabulary;

        public Tokenizer(List<string> vocabulary)
        {
            _vocabulary = vocabulary;
        }

        public List<(string Token, int VocabularyIndex, long SegmentIndex)> Tokenize(params string[] texts)
        {
            IEnumerable<string> tokens = new string[] { Tokens.Classification };

            foreach (var text in texts)
            {
                tokens = tokens.Concat(TokenizeSentence(text));
                tokens = tokens.Concat(new string[] { Tokens.Separation });
            }

            var tokenAndIndex = tokens
                .SelectMany(TokenizeSubwords)
                .ToList();

            var segmentIndexes = SegmentIndex(tokenAndIndex);

            return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex) 
                                => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();
        }

        public List<string> Untokenize(List<string> tokens)
        {
            var currentToken = string.Empty;
            var untokens = new List<string>();
            tokens.Reverse();

            tokens.ForEach(token =>
            {
                if (token.StartsWith("##"))
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
                else
                {
                    currentToken = token + currentToken;
                    untokens.Add(currentToken);
                    currentToken = string.Empty;
                }
            });

            untokens.Reverse();

            return untokens;
        }

        public IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == Tokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
        {
            if (_vocabulary.Contains(word))
            {
                return new (string, int)[] { (word, _vocabulary.IndexOf(word)) };
            }

            var tokens = new List<(string, int)>();
            var remaining = word;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                var prefix = _vocabulary.Where(remaining.StartsWith)
                    .OrderByDescending(o => o.Count())
                    .FirstOrDefault();

                if (prefix == null)
                {
                    tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));

                    return tokens;
                }

                remaining = remaining.Replace(prefix, "##");

                tokens.Add((prefix, _vocabulary.IndexOf(prefix)));
            }

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                tokens.Add((Tokens.Unknown, _vocabulary.IndexOf(Tokens.Unknown)));
            }

            return tokens;
        }

        private IEnumerable<string> TokenizeSentence(string text)
        {
            // remove spaces and split the , . : ; etc..
            return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
                .SelectMany(o => o.SplitAndKeep(".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()))
                .Select(o => o.ToLower());
        }
    }
}