using System;
using MathNet.Numerics.LinearAlgebra;

namespace neural_network
{
    class Program
    {
        static void Main(string[] args)
        {
            // var xTrain = new Matrix<double>[]
            // {
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {0, 0}}),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {0, 1}}),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {1, 0}}),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {1, 1}})
            // };

            // var yTrain = new Matrix<double>[]
            // {
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {0} }),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {1} }),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {1} }),
            //     Matrix<double>.Build.DenseOfArray(new double[,] { {0} })
            // };

            var xTrain = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[,]
                {
                    {0, 0},
                    {0, 1},
                    {1, 0},
                    {1, 1}
                })
            };

            var yTrain = new Matrix<double>[]
            {
                Matrix<double>.Build.DenseOfArray(new double[,] { 
                    {0},
                    {1},
                    {1},
                    {0}
                })
            };

            NeuralNetwork neuralNetwork = new NeuralNetwork();

            int batchSize = xTrain[0].RowCount;

            neuralNetwork.Add(new FCLayer(2, 3, batchSize));
            neuralNetwork.Add(new ActivationLayer(new TanhActivationFunction()));
            neuralNetwork.Add(new FCLayer(3, 1, batchSize));
            neuralNetwork.Add(new ActivationLayer(new TanhActivationFunction()));

            neuralNetwork.Use(new MSELossFunction());
            neuralNetwork.Fit(xTrain, yTrain, 1000, 0.1);

            var result = neuralNetwork.Predict(xTrain);
            for (int i = 0; i < result.Length; i++)
            {
                Console.WriteLine($"pred: {result[i].ToString()}");
            }
        }
    }
}
