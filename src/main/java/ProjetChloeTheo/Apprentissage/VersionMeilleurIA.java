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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author chloe
 */
public class VersionMeilleurIA {
    
     private static MultiLayerNetwork model;
     
     // Fonction main orchestrant tout le processus
    public static void main(String[] args) {
        try {
            String csvFilePath = "C:\\temp\\noirs8000.csv"; // Chemin vers votre fichier CSV

            DataSet dataset = createDataset(csvFilePath);
            System.out.println("Dataset created with " + dataset.numExamples() + " examples.");

            // Split du dataset en ensembles d'entraînement et de test
            dataset.shuffle();
            int trainSize = (int) (dataset.numExamples() * 0.8);
            int testSize = dataset.numExamples() - trainSize;
            DataSet trainData = dataset.splitTestAndTrain(trainSize).getTrain();
            DataSet testData = dataset.splitTestAndTrain(testSize).getTest();

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), 128);
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), 128);

            // Paramètres du modèle
            int seed = 123;
            double learningRate = 0.001;

            // Création du modèle
            model = createModel(seed, learningRate);

            // Entraînement du modèle
            trainModel(model, trainIterator);

            // Enregistrement du modèle
            saveModel(model, "othello-mlp-model.zip");

            // Évaluation du modèle
            evaluateModel(model, testIterator);

        } catch (IOException | IllegalArgumentException e) {
            e.printStackTrace();
        }
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
                if (values.length != 67) { // Ajusté pour 67 colonnes (64 entrées + 1 sortie + 2 qui ne sont pas utiles)
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
    
     // Fonction de création du modèle de réseau neuronal
    public static MultiLayerNetwork createModel(int seed, double learningRate) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(learningRate, 0.9))
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(64) // Nombre d'entrées (64 cases du plateau Othello)
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new DenseLayer.Builder()
                .nOut(256)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Utilisation de la MSE pour la régression
                .nOut(1) // Une seule sortie pour la probabilité de victoire
                .activation(Activation.IDENTITY) // Activation identité pour la régression
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }
    
    // Fonction d'entraînement du modèle
    public static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator) {
        System.out.println("Training model...");
        for (int i = 0; i < 1; i++) { // Nombre d'époques = 1 pour simplification
            model.fit(trainIterator);
            System.out.println("Completed epoch " + (i + 1));
        }
    }
    
     // Fonction d'enregistrement du modèle
    public static void saveModel(MultiLayerNetwork model, String modelPath) throws IOException {
        System.out.println("Saving the model...");
        model.save(new File(modelPath));
        System.out.println("Model saved to " + modelPath);
    }
    
    // Fonction pour charger un modèle enregistré à partir d'un fichier ZIP
    public static MultiLayerNetwork loadModel(String modelPath) throws IOException {
        System.out.println("Loading model from: " + modelPath);
        MultiLayerNetwork loadedModel = MultiLayerNetwork.load(new File(modelPath), true);
        System.out.println("Model loaded successfully.");
        return loadedModel;
    }
    
    // Fonction d'évaluation du modèle
    public static void evaluateModel(MultiLayerNetwork model, DataSetIterator testIterator) {
        System.out.println("Evaluating model...");
        RegressionEvaluation eval = new RegressionEvaluation();

        while (testIterator.hasNext()) {
            DataSet t = testIterator.next();
            if (t.getFeatures().isEmpty() || t.getLabels().isEmpty()) {
                System.out.println("Skipping empty DataSet.");
                continue;
            }

            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();

            if (features.rank() != 2 || labels.rank() != 2) {
                System.out.println("Skipping DataSet with invalid dimensions.");
                continue;
            }

            INDArray predicted = model.output(features, false);

            if (predicted.isEmpty()) {
                System.out.println("Skipping empty prediction.");
                continue;
            }

            eval.eval(labels, predicted);
        }

        System.out.println(eval.stats());
    }
    
    // Exemple pour utiliser le modèle pour faire une prédiction avec de nouvelles données
    public static void makePrediction(MultiLayerNetwork model, INDArray newInput) {
        INDArray prediction = model.output(newInput);
        System.out.println("Prediction: " + prediction);
    }


     
     
    
    
    
}
