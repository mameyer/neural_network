using System;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System.Collections.Generic;

namespace neural_network
{
    class Program
    {
        public static void Xor(bool batch = true)
        {
            Matrix<double>[] xTrain, yTrain;
            
            if (batch)
            {
                xTrain = new Matrix<double>[]
                {
                    Matrix<double>.Build.DenseOfArray(new double[,]
                    {
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}
                    })
                };

                yTrain = new Matrix<double>[]
                {
                    Matrix<double>.Build.DenseOfArray(new double[,] { 
                        {0},
                        {1},
                        {1},
                        {0}
                    })
                };
            }
            else
            {
                xTrain = new Matrix<double>[]
                {
                    Matrix<double>.Build.DenseOfArray(new double[,] { {0, 0}}),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {0, 1}}),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {1, 0}}),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {1, 1}})
                };

                yTrain = new Matrix<double>[]
                {
                    Matrix<double>.Build.DenseOfArray(new double[,] { {0} }),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {1} }),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {1} }),
                    Matrix<double>.Build.DenseOfArray(new double[,] { {0} })
                };
            }


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

        public static void Mnist(int maxSamples = 1000)
        {
            NeuralNetwork neuralNetwork = new NeuralNetwork();

            int batchSize = 1;

            neuralNetwork.Add(new FCLayer(28*28, 100, batchSize));
            neuralNetwork.Add(new ActivationLayer(new TanhActivationFunction()));
            neuralNetwork.Add(new FCLayer(100, 50, batchSize));
            neuralNetwork.Add(new ActivationLayer(new TanhActivationFunction()));
            neuralNetwork.Add(new FCLayer(50, 10, batchSize));
            neuralNetwork.Add(new ActivationLayer(new TanhActivationFunction()));

            var xTrain = new List<Matrix<double>>();
            var yTrain = new List<Matrix<double>>();

            int samples = 0;
            foreach (var image in mnist.MnistLoader.ReadTrainingData())
            {
                xTrain.Add(image.Data); 
                yTrain.Add(image.Label);

                samples++;
                if (samples >= maxSamples) break;
            }

            neuralNetwork.Use(new MSELossFunction());
            neuralNetwork.Fit(xTrain.ToArray(), yTrain.ToArray(), 50, 0.1);
        }

        static void Main(string[] args)
        {
            Console.WriteLine("xor..");
            Xor();
            Console.WriteLine();

            Console.WriteLine("mnist..");
            Mnist();
        }
    }
}
