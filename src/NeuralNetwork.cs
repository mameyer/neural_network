using System;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace neural_network
{
    public class NeuralNetwork
    {
        public NeuralNetwork()
        {
            this.Layers = new List<Layer>();
        }

        public List<Layer> Layers { get; set; }
        public ILossFunction LossFunction { get; set; }

        public void Add(Layer layer)
        {
            this.Layers.Add(layer);
        }

        public void Use(ILossFunction lossFunction)
        {
            this.LossFunction = lossFunction;
        }

        public Matrix<double>[] Predict(Matrix<double>[] inputData)
        {
            var samples = new Matrix<double>[inputData.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                var output = inputData[i];
                foreach (Layer layer in this.Layers)
                {
                    output = layer.ForwardPropagation(output);
                }
                samples[i] = output;
            }

            return samples;
        }

        public void Fit(Matrix<double>[] xTrain, Matrix<double>[] yTrain, int epochs, double learningRate)
        {
            int samples = xTrain.Length;
            if (samples <= 0) return;

            int batchSize = xTrain[0].RowCount;
            Console.WriteLine($"epochs: {epochs}");
            Console.WriteLine($"learning rate: {learningRate}");
            Console.WriteLine($"batch size: {batchSize}");
            Console.WriteLine($"samples: {samples}");

            for (int i = 0; i < epochs; i++)
            {
                double err = 0;
                for (int j = 0; j < samples; j++)
                {
                    var output = xTrain[j];
                    foreach (var layer in this.Layers)
                    {
                        output = layer.ForwardPropagation(output);
                    }

                    for (int k = 0; k < output.RowCount; k++)
                    {
                        err += this.LossFunction.Loss(yTrain[j].Row(k).ToRowMatrix(), output.Row(k).ToRowMatrix());
                    }

                    var error = this.LossFunction.LossPrime(yTrain[j], output);
                    for (int k = this.Layers.Count - 1; k >= 0; k--)
                    {
                        error = this.Layers.ElementAt(k).BackwardPropagation(error, learningRate);
                    }
                }

                err /= samples;

                if (i == 0 || i == epochs -1 || i % 50 == 0)
                {
                    Console.WriteLine($"epoch {i}/{epochs}: err={err}");
                }
            }
        }
    }
}