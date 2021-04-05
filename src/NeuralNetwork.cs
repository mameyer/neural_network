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

        private void PrintNetworkLayout()
        {
            Console.WriteLine("network layout:");
            for (int i = 0; i < this.Layers.Count(); i++)
            {
                var layer = this.Layers.ElementAt(i);

                if (layer is ActivationLayer activationLayer)
                {
                    Console.WriteLine($"layer {i}: activation");
                    continue;
                }

                if (layer is FCLayer fcLayer)
                {
                    Console.WriteLine($"layer {i}: fully connected");
                    Console.WriteLine($"  in  := {fcLayer.Input.RowCount}x{fcLayer.Input.ColumnCount}");
                    Console.WriteLine($"  w   := {fcLayer.Weights.RowCount}x{fcLayer.Weights.ColumnCount}");
                    Console.WriteLine($"  out := {fcLayer.Output.RowCount}x{fcLayer.Output.ColumnCount}");
                    Console.WriteLine($"  b   := {fcLayer.Bias.RowCount}x{fcLayer.Bias.ColumnCount}");
                }
            }
        }

        public void Fit(Matrix<double>[] xTrain, Matrix<double>[] yTrain, int epochs, double learningRate,
            int printEveryEpoch = 1)
        {
            int samples = xTrain.Length;
            if (samples <= 0 || yTrain.Length != samples)
            {
                Console.WriteLine("got no samples OR x/y length mismatch..");
                return;
            }

            int batchSize = xTrain[0].RowCount;
            Console.WriteLine($"epochs: {epochs}");
            Console.WriteLine($"learning rate: {learningRate}");
            Console.WriteLine($"batch size: {batchSize}");
            Console.WriteLine();

            Console.WriteLine($"samples: {samples}");
            Console.WriteLine($"  x: {xTrain.Length}");
            Console.WriteLine($"    size: {xTrain[0].RowCount}, {xTrain[0].ColumnCount}");
            Console.WriteLine($"  y: {yTrain.Length}");
            Console.WriteLine($"    size: {yTrain[0].RowCount}, {yTrain[0].ColumnCount}");
            Console.WriteLine("--------------");
            Console.WriteLine();

            PrintNetworkLayout();
            Console.WriteLine("--------------");
            Console.WriteLine();

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

                if (i == 0 || i == epochs -1 || i % printEveryEpoch == 0)
                {
                    Console.WriteLine($"epoch {i}/{epochs}: err={err}");
                }
            }
        }
    }
}