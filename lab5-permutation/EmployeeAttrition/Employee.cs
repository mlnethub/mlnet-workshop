using Microsoft.ML.Data;

namespace EmployeeAttrition
{
    public class Employee
    {
        [LoadColumn(0)]
        public float Age { get; set; }
    
        [LoadColumn(1)]
        public bool Attrition { get; set; }

        [LoadColumn(2)]
        public string BusinessTravel { get; set; }

        [LoadColumn(3)]
        public float DailyRate { get; set; }

        [LoadColumn(4)]
        public string Department { get; set; }

        [LoadColumn(5)]
        public float DistanceFromHome { get; set; }

        [LoadColumn(6)]
        public float Education { get; set; }

        [LoadColumn(7)]
        public string EducationField { get; set; }

        [LoadColumn(8)]
        public float EmployeeCount { get; set; }

        // Column 9: Do not include EmployeeNumber

        [LoadColumn(10)]
        public float EnvironmentSatisfaction { get; set; }

        [LoadColumn(11)]
        public string Gender { get; set; }

        [LoadColumn(12)]
        public float HourlyRate { get; set; }

        [LoadColumn(13)]
        public float JobInvolvement { get; set; }

        [LoadColumn(14)]
        public string JobLevel { get; set; }

        [LoadColumn(15)]
        public string JobRole { get; set; }

        [LoadColumn(16)]
        public float JobSatisfaction { get; set; }

        [LoadColumn(17)]
        public string MaritalStatus { get; set; }

        [LoadColumn(18)]
        public float MonthlyIncome { get; set; }

        [LoadColumn(19)]
        public float MonthlyRate { get; set; }

        [LoadColumn(20)]
        public float NumCompaniesWorked { get; set; }

        // Column 21: Do not include Over18

        [LoadColumn(22)]
        public string OverTime { get; set; }

        [LoadColumn(23)]
        public float PercentSalaryHike { get; set; }

        [LoadColumn(24)]
        public float PerformanceRating { get; set; }

        [LoadColumn(25)]
        public float RelationshipSatisfaction { get; set; }

        // Column 26: Do not include StandardHours

        [LoadColumn(27)]
        public float StockOptionLevel { get; set; }

        [LoadColumn(28)]
        public float TotalWorkingYears { get; set; }

        [LoadColumn(29)]
        public float TrainingTimesLastYear { get; set; }

        [LoadColumn(30)]
        public float WorkLifeBalance { get; set; }

        [LoadColumn(31)]
        public float YearsAtCompany { get; set; }

        [LoadColumn(32)]
        public float YearsInCurrentRole { get; set; }

        [LoadColumn(33)]
        public float YearsSinceLastPromotion { get; set; }

        [LoadColumn(34)]
        public float YearsWithCurrManager { get; set; }
    }
}