using System;
using System.Collections.Generic;
using System.Linq;

namespace BertMlNet.Extensions
{
    public static class SoftmaxEnumerableExtension
    {
        public static IEnumerable<(T Item, float Probability)> Softmax<T>(
                                            this IEnumerable<T> collection, 
                                            Func<T, float> scoreSelector)
        {
            var maxScore = collection.Max(scoreSelector);
            var sum = collection.Sum(r => Math.Exp(scoreSelector(r) - maxScore));

          return collection.Select(r => (r, (float)(Math.Exp(scoreSelector(r) - maxScore) / sum)));
        }
    }
}