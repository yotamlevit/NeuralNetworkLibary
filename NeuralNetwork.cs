using System;
using System.Collections.Generic;
using System.Text;
using GraphLibary;
using System.Linq;
static class Constants
{
    public const double SigmoidAPara = 1;
}

namespace NeuralNetworkLibary
{
    public class NeuralNetwork : INeuralNetwork
    {
        private DirectedGraph<string, double> DGraph;
        private double[][][] weights;
        private double[][] layes;

        //Constractor
        //1) dGraph - a Directed Graph that will represent the NN
        //2) layerCount - an array in length of the number of layers in the NN/ the graph
        //                each sell will contain the number of activations
        public NeuralNetwork(DirectedGraph<string, double> dGraph, int[] layerCount)
        {
            this.DGraph = dGraph;
            this.initMatrixes(layerCount);
        }

        //Update the NN same as constactor
        //1) dGraph - a Directed Graph that will represent the NN
        //2) layerCount - an array in length of the number of layers in the NN/ the graph
        //   
        public void UpdateNeuralNetwork(DirectedGraph<string, double> dGraph, int[] layerCount)
        {
            this.DGraph = dGraph;
            this.initMatrixes(layerCount);
        }

        //init the matrixes
        // init the weights metrix and the layes metrix that contain the activations
        //1) layerCount - an array in length of the number of layers in the NN/ the graph
        private void initMatrixes(int[] layerCount)
        {
            if (layerCount == null)
                throw new ArgumentException();
            this.initActivationsMetrix(layerCount);
            this.initWeightsMetrix();
        }

        //init the layes metrix that contain the activations
        //1) layerCount - an array in length of the number of layers in the NN/ the graph
        private void initActivationsMetrix(int[] layerCount)
        {
            if (layerCount == null)
                throw new ArgumentNullException();
            this.layes = new double[layerCount.Length][];
            for (int i = 0; i < layerCount.Length; i++)
            {
                this.layes[i] = new double[layerCount[i]];
            }
        }

        //init the weights metrix
        private void initWeightsMetrix()
        {
            this.weights = new double[this.layes.Length - 1][][];
            for (int i = 0; i < this.weights.Length; i++)
            {
                this.weights[i] = new double[this.layes[i + 1].Length][];
                for (int j = 0; j < this.weights[i].Length; j++)
                {
                    this.weights[i][j] = new double[this.layes[i].Length];
                }
            }
            List<IPairValue<string>> EdgeSet = this.DGraph.GetEdgeSet().ToList();
            IPairValue<string> edge;
            int indexWeight = 0;
            for (int i = 0; i < this.weights.Length; i++)
            {
                for (int j = 0; j < this.weights[i].Length; j++)
                {
                    for (int k = 0; k < this.weights[i][j].Length; k++)
                    {
                        edge = EdgeSet[indexWeight++];
                        this.weights[i][j][k] = this.DGraph.GetWeight(edge.GetFirst(), edge.GetSecond());
                    }
                }
            }
        }

        //return the output from the all NN with a given input
        //1)inputs - an array if doubles that contain the givin inputs
        public int NextMove(double[] inputs)
        {
            Input(inputs);
            FeedForward();
            return Output();
        }

        //Input
        //place the inputs in the first layer of the activations (the inputs ones)
        private void Input(double[] inputs)
        {   
            for (int i = 0; i < inputs.Length; i++)
            {
                this.layes[0][i] = SigmoidInputFunction(inputs[i]);
                // this.layes[0][i] = (inputs[i]);
            }
        }

        //Output
        //return the number of the output node
        private int Output()
        {
            int maxIndex = 0;
            for (int i = 0; i < this.layes[this.layes.Length - 1].Length; i++)
            {
                if (this.layes[this.layes.Length - 1][i] > this.layes[this.layes.Length - 1][maxIndex])
                    maxIndex = i;
            }
            return maxIndex;
        }

        //the math function for Sigmoid Funciton
        //the function gaets a nubmer and return its value after the sigmoid funtion
        // f(x) = 1/(1+e^(-a*x)) ; a = Constants.SigmoidAPara
        //1)x - a double number that cintain the x value
        private double SigmoidFunction(double x)
        {
            double ret = 1 / (1 + Math.Pow(Math.E, (-1 * Constants.SigmoidAPara * (x+2))));
            return ret;
        }

        private double SigmoidInputFunction(double x)
        {
            double ret = 1 / (1 + Math.Pow(Math.E, (-1 * Constants.SigmoidAPara * (x-25))));
            return ret;
        }

        // mul the weight metrix with the activations
        //saves the answer in the next layer
        private void WeightsMulActivation(double[][] curr_weights, double[] SrcLayer, double[] DesLayer)
        {
            for (int i = 0; i < curr_weights.Length; i++)
            {
                for (int j = 0; j < curr_weights[i].Length; j++)
                {
                    DesLayer[i] += curr_weights[i][j] * SrcLayer[j];
                }
                DesLayer[i] = SigmoidFunction(DesLayer[i]);
                //DesLayer[i] = (DesLayer[i]);

            }
        }

        //feed the input forward in the inputs to the output layer
        private void FeedForward()
        {
            for (int i = 0; i < this.weights.Length; i++)
            {
                WeightsMulActivation(this.weights[i], this.layes[i], this.layes[i + 1]);
            }
        }

        //print the metrixes
        public void PrintMetrix()
        {
            for (int i = 0; i < this.layes.Length; i++)
            {
                Console.WriteLine("Layer:" + i);
                for (int j = 0; j < this.layes[i].Length; j++)
                {
                    Console.Write(this.layes[i][j] + ", ");
                }
                Console.WriteLine();
                Console.WriteLine();
            }
            Console.WriteLine();
            for (int i = 0; i < this.weights.Length; i++)
            {
                Console.WriteLine("Layer:" + i);
                for (int j = 0; j < this.weights[i].Length; j++)
                {
                    Console.WriteLine();
                    for (int k = 0; k < this.weights[i][j].Length; k++)
                    {
                        Console.Write(this.weights[i][j][k] + ", ");
                    }
                }
                Console.WriteLine();
                Console.WriteLine();
            }
        }
    }
}
