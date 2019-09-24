using System;
using Microsoft.ML.Data;

namespace MovieRecByFFM.DataStructures
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public string userId;

        [LoadColumn(1)]
        public string movieId;

        [LoadColumn(2)]
        public float rating;
    }
}