/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author chloe
 */

import java.io.BufferedReader;
import java.io.FileReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

public class CNN {
   
    private static MultiLayerNetwork model;

    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs5000.csv";
            
            // Création du dataset
            DataSet dataset = createDataset(csvFilePath);
            System.out.println("Dataset created with " + dataset.numExamples() + " examples.");

            // Reshape des données pour le format CNN (batch, channels, height, width)
            reshapeDataForCNN(dataset);

            // Split train/test
            dataset.shuffle(123);
            SplitTestAndTrain splits = dataset.splitTestAndTrain(0.8);
            DataSet trainData = splits.getTrain();
            DataSet testData = splits.getTest();

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), 32);
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), 32);

            // Création et entraînement du modèle CNN
            model = createCNNModel();
            trainCNNModel(model, trainIterator, testIterator, 30);

            // Sauvegarde et évaluation
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn-model.zip";
            ModelSerializer.writeModel(model, modelPath, true);
            evaluateModel(modelPath, testIterator);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
       // Méthode améliorée d'évaluation
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        RegressionEvaluation eval = new RegressionEvaluation();

        // Variables pour le suivi des prédictions
        int totalPredictions = 0;
        int correctPredictions = 0;
        double threshold = 0.5; // Seuil pour classification binaire

        while (testIterator.hasNext()) {
            DataSet batch = testIterator.next();
            INDArray features = batch.getFeatures();
            INDArray labels = batch.getLabels();
            INDArray predictions = model.output(features);

            // Évaluation des métriques de régression
            eval.eval(labels, predictions);

            // Calcul de la précision pour la classification
            for(int i = 0; i < predictions.length(); i++) {
                totalPredictions++;
                double predicted = predictions.getDouble(i);
                double actual = labels.getDouble(i);

                // On considère une prédiction correcte si elle est du bon côté du seuil
                if((predicted >= threshold && actual >= threshold) ||
                   (predicted < threshold && actual < threshold)) {
                    correctPredictions++;
                }
            }
        }

        // Affichage des résultats détaillés
        System.out.println("\nRésultats de l'évaluation:");
        System.out.println(eval.stats());
        System.out.printf("Précision de classification: %.2f%% (%d/%d)%n", 
            (100.0 * correctPredictions / totalPredictions),
            correctPredictions, totalPredictions);
    }

    private static void reshapeDataForCNN(DataSet dataset) {
        // Reshape des features de (N, 64) à (N, 1, 8, 8)
        INDArray features = dataset.getFeatures();
        INDArray labels = dataset.getLabels();
        
        int numExamples = (int) features.size(0);
        INDArray reshapedFeatures = features.reshape(numExamples, 1, 8, 8);
        
        dataset.setFeatures(reshapedFeatures);
        dataset.setLabels(labels);
    }

    private static MultiLayerNetwork createCNNModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            // Première couche convolutive
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                .nIn(1)
                .nOut(32)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            // Deuxième couche convolutive
            .layer(1, new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            // Pooling
            .layer(2, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(1, 1)
                .build())
            // Dense layer
            .layer(3, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            // Output layer
            .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nOut(1)
                .activation(Activation.SIGMOID)
                .build())
            .setInputType(InputType.convolutional(8, 8, 1))
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        return model;
    }

    private static void trainCNNModel(MultiLayerNetwork model, DataSetIterator trainIterator, 
                                    DataSetIterator testIterator, int nEpochs) {
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            model.fit(trainIterator);
            
            // Évaluation périodique
            RegressionEvaluation eval = new RegressionEvaluation(1);
            while (testIterator.hasNext()) {
                DataSet testSet = testIterator.next();
                INDArray predictions = model.output(testSet.getFeatures());
                eval.eval(testSet.getLabels(), predictions);
            }
            
            System.out.printf("\nÉpoque %d terminée:%n", epoch + 1);
            System.out.printf("MSE: %.4f%n", eval.averageMeanSquaredError());
            System.out.printf("MAE: %.4f%n", eval.averageMeanAbsoluteError());
            System.out.printf("RMSE: %.4f%n", Math.sqrt(eval.averageMeanSquaredError()));
            System.out.printf("Correlation: %.4f%n", eval.pearsonCorrelation(0));
            
            trainIterator.reset();
            testIterator.reset();
        }
    }

    // Cette méthode permet de convertir un plateau Othello en entrée pour le CNN
    public static INDArray boardToCNNInput(SituationOthello situation) {
        double[] board = situation.getBoardAsArray();
        return Nd4j.create(board).reshape(1, 1, 8, 8);
    }
    
    // Fonction de création du dataset à partir d'un fichier CSV
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>();
        List<INDArray> outputList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) {
            String line;
            int lineNumber = 0;
            while ((line = br.readLine()) != null) {
                lineNumber++;
                String[] values = line.split(",");
                if (values.length != 65) { // Ajusté pour 65 colonnes (64 entrées + 1 sortie)
                    System.err.println("Line " + lineNumber + " has " + values.length + " columns: " + line);
                    continue; // Sauter cette ligne
                }

                // Créer l'entrée INDArray avec 64 valeurs
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

                // Créer la sortie INDArray avec 1 valeur
                INDArray output = Nd4j.zeros(1, 1);
                try {
                    output.putScalar(0, Double.parseDouble(values[64]));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number at line " + lineNumber + " column 64: " + values[64]);
                    continue;
                }
                
                // Ajouter les entrées et sorties validées à leurs listes respectives
                inputList.add(input);
                outputList.add(output);
            }
        }
        
        if (inputList.isEmpty() || outputList.isEmpty()) {
            throw new IllegalArgumentException("No valid data found in the CSV file.");
        }

        // Empiler les entrées et sorties en un seul INDArray
        INDArray input = Nd4j.vstack(inputList);
        INDArray output = Nd4j.vstack(outputList);
        
        // Vérification de la forme des tableaux
        if (input.rank() != 2 || output.rank() != 2) {
            throw new IllegalStateException("Input or output has invalid shape. Rank: " + input.rank() + ", " + output.rank());
        }

        DataSet dataset = new DataSet(input, output);

        // Normalisation du dataset
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataset);
        normalizer.transform(dataset);

        return dataset;
    }
    
}

