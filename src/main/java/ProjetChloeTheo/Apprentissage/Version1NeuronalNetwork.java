/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */


package ProjetChloeTheo.Apprentissage;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
 /*
 * @author chloe
 */
public class Version1NeuronalNetwork {
    static MultiLayerNetwork model;
        
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                String[] values = line.split(",");
                if (values.length != 67) { // Adjusted to 65 columns (64 inputs + 1 output)
                    System.err.println("Line " + lineNumber + " has " + values.length + " columns: " + line);
                    continue; // Skip this line
                }

                // Create input INDArray with 64 values
                INDArray input = Nd4j.zeros(1, 64);
                boolean validInput = true;
                for (int i = 0; i < 64; i++) {
                    try {
                        input.putScalar(i, Double.parseDouble(values[i]));
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid number at line " + lineNumber + " column " + i + ": " + values[i]);
                        validInput = false;
                        break;
                    }
                }
                if (!validInput) continue;

                // Create output INDArray with 1 value
                INDArray output = Nd4j.zeros(1, 1);
                try {
                    output.putScalar(0, Double.parseDouble(values[64]));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number at line " + lineNumber + " column 64: " + values[64]);
                    continue;
                }
                
                 // Check if input and output are valid before adding them
                if (input.columns() != 64) {
                    System.err.println("Invalid input dimensions at line " + lineNumber);
                    continue;
                }
                if (output.columns() != 1) {
                    System.err.println("Invalid output dimensions at line " + lineNumber);
                    continue;
                }

                inputList.add(input);
                outputList.add(output);
            }
        }
        
        if (inputList.isEmpty() || outputList.isEmpty()) {
            throw new IllegalArgumentException("No valid data found in the CSV file.");
        }

        // Stack all input and output INDArrays into single INDArrays
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);
        
         // Check if the final arrays have correct shapes
        if (input.rank() != 2 || output.rank() != 2) {
            throw new IllegalStateException("Input or output has invalid shape. Rank: " + input.rank() + ", " + output.rank());
        }

        DataSet dataset = new DataSet(input, output);

        // Optional: Normalize the dataset
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);

        return dataset;
    }

    public static void main(String[] args) {
        try {
            String csvFilePath = "C:\\temp\\noirs8000.csv"; // Path to your CSV file

            DataSet dataset = createDataset(csvFilePath);
            System.out.println("Dataset created with " + dataset.numExamples() + " examples.");

            // Split dataset into training and test sets
            dataset.shuffle();
            int trainSize = (int) (dataset.numExamples() * 0.8);
            int testSize = dataset.numExamples() - trainSize;
            DataSet trainData = dataset.splitTestAndTrain(trainSize).getTrain();
            DataSet testData = dataset.splitTestAndTrain(testSize).getTest();

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), 128);
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), 128);

            int seed = 123;
            double learningRate = 0.001;

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.9))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                    .nIn(64) // Number of input neurons (64 squares of Othello board)
                    .nOut(256)
                    .activation(Activation.RELU)
                    .build())
                .layer(new DenseLayer.Builder()
                    .nOut(256)
                    .activation(Activation.RELU)
                    .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Use MSE for regression
                    .nOut(1) // One output neuron for the win probability
                    .activation(Activation.IDENTITY) // Identity activation for regression
                    .build())
                .build();

            model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));

            System.out.println("Training model...");
            for (int i = 0; i < 10; i++) { // numEpochs = 10
                model.fit(trainIterator);
                System.out.println("Completed epoch " + (i + 1));
            }

            System.out.println("Evaluating model...");
            RegressionEvaluation eval = new RegressionEvaluation();
            while (testIterator.hasNext()) {
                DataSet t = testIterator.next();
                INDArray features = t.getFeatures();
                INDArray labels = t.getLabels();
                INDArray predicted = model.output(features, false);
                eval.eval(labels, predicted);
            }
            System.out.println(eval.stats());

            model.save(new File("othello-mlp-model.zip"));
            System.out.println("Model training complete.");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        }
    }
   }