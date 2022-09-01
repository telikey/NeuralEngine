using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace NeuralEngine
{
    public interface IModelInput
    {

        string Label { get; set; }
    }

    public interface IModelOutput
    {
        public string PredictedLabelValue { get; set; }

        public float[] Score { get; set; }
    }

    public class NeuralEngine<TInput, TOutput> where TInput : class, IModelInput, new() where TOutput : class, IModelOutput, new()
    {
        MLContext mlContext { get; set; }

        dynamic PipeLine { get; set; }

        ITransformer? Model { get; set; }

        PredictionEngine<TInput, TOutput> PredictionEngine { get; set; }

        public NeuralEngine()
        {
            mlContext = new MLContext();

            var props = typeof(TInput).GetProperties().Where(x => x.Name != "Label").Select(x=>x.Name).Take(700).ToArray();

            if (props != null)
            {
                var dataProcessPipeline = mlContext.Transforms
                    .Concatenate("Features", props)
                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                    .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", featureColumnName: "Features"))
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                    .AppendCacheCheckpoint(mlContext);

                PipeLine = dataProcessPipeline;

            }
        }

        public void Train(TInput[] trainInput)
        {
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainInput);
            Model = PipeLine.Fit(trainingData);
            PredictionEngine=mlContext.Model.CreatePredictionEngine<TInput, TOutput>(Model);
        }

        public TOutput? Think(TInput input)
        {
            if (PredictionEngine != null)
            {
                var answ = PredictionEngine.Predict(input);
                return answ;
            }
            return null;
        }
    }
}