using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuralEngine
{
    public interface IModelInput
    {

        [ColumnName("Class"), LoadColumn(2)]
        public bool Class { get; set; }
    }

    public interface IModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }

    public class NeuralEngine<TInput, TOutput> where TInput : class, new() where TOutput : class, new()
    {
        MLContext mlContext { get; set; }

        dynamic PipeLine { get; set; }

        ITransformer? Model { get; set; }

        public NeuralEngine(int numberOfIterations = 10000, double learningRate = 0.001)
        {
            mlContext = new MLContext();

            var props = typeof(TInput).GetProperties().Where(x => x.Name != "Class").Select(x => x.Name).ToArray();

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", props);
            var trainer = mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: "Class", featureColumnName: "Features", numberOfIterations: numberOfIterations, learningRate: learningRate);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            PipeLine = trainingPipeline;
        }

        public void Train(TInput[] trainInput)
        {
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainInput);
            Model = PipeLine.Fit(trainingData);
        }

        public TOutput? Think(TInput input)
        {
            if (Model != null)
            {
                var predEngine = mlContext.Model.CreatePredictionEngine<TInput, TOutput>(Model);

                var answ = predEngine.Predict(input);
                return answ;
            }
            return null;
        }
    }
}