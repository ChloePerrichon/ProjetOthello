/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */


package ProjetChloeTheo.Apprentissage;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
/**
 *
 * @author chloe
 */
public class Version1NeuronalNetwork {
    
    static MultiLayerNetwork model;

    public static void main(String[] args) throws IOException, InterruptedException {
        int batchSize = 10000;
        int seed = 123;
        int numEpochs = 10;
        double learningRate = 0.001;

        // Load dataset from CSV
        String csvFilePath = "C:\\temp\\noirs8000.csv"; // Remplacez par le chemin réel vers votre fichier CSV
        DataSet allData = readCSVDataset(csvFilePath, batchSize);

        // Normalize the dataset
        DataNormalization scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fit(allData);
        scaler.transform(allData);

        // Split the dataset into training and testing sets
        DataSet trainData = allData.splitTestAndTrain(0.8).getTrain();
        DataSet testData = allData.splitTestAndTrain(0.8).getTest();

       // List<DataSet> trainList = trainData.asList();
      //  List<DataSet> testList = testData.asList();

      //  DataSetIterator trainIter = new ListDataSetIterator(trainList, batchSize);
       // DataSetIterator testIter = new ListDataSetIterator(testList, batchSize);

        // Configure the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(learningRate, 0.9))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(trainData.numInputs())
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(trainData.numOutcomes())
                .activation(Activation.SOFTMAX)
                .build())
            .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        System.out.println("Training model...");
        for (int i = 0; i < numEpochs; i++) {
          //  model.fit(trainIter);
            System.out.println("Completed epoch " + (i + 1));
        }

        // Evaluate the model
        System.out.println("Evaluating model...");
       // Evaluation eval = model.evaluate(testIter);
      //  System.out.println(eval.stats());

        // Save the model
        model.save(new File("othello-mlp-model.zip"));
        System.out.println("Model training complete.");
    }

    private static DataSet readCSVDataset(String csvFilePath, int batchSize) throws IOException, InterruptedException {
        int labelIndex = 64;  // Index of the label column (change this based on your CSV structure)
        int numClasses = 3;   // Number of classes for the output (e.g., 3 for win/lose/draw)

        // Create a CSVRecordReader with default settings
        CSVRecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new File(csvFilePath)));

        // Create an iterator from the record reader
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        // Collect all examples in a list
        List<DataSet> dataSetList = new ArrayList<>();
        while (iterator.hasNext()) {
            dataSetList.add(iterator.next());
        }

        // Concatenate all the DataSet objects into a single DataSet
        DataSet allData = DataSet.merge(dataSetList);
        return allData;
    }
}