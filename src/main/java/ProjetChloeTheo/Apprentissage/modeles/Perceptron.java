package ProjetChloeTheo.Apprentissage.modeles;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author chloe
 */
public class Perceptron {
    private static MultiLayerNetwork model;
    
    
    public static void main(String[] args) {
        try {
            String csvFilePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba5000CNN-PERCEPTRON.csv"; // chemin du csv sur lequel on veut entrainer le modèle
            
            // Paramètres du modèle
            int seed = 123; //graine de reproductibilité
            double learningRate = 0.001; // taux d'apprentissage 
            int numEpochs = 30; //nombre d'époques: nombre de fois que le modèle voit l'ensemble des données 
            int batchSize=128;
            
            // Création et préparation du dataset
            DataSet dataset = createDataset(csvFilePath); // On crée un dataset en INDArray à partir du csv contenant les situations et les probas
            System.out.println("Dataset created with " + dataset.numExamples() + " examples."); 

            // Division du dataset en deux ensembles: un d'entrainement et un de test
            dataset.shuffle(seed); //mélange de façon aléatoire les lignes du Dataset (garde bien les situations intactes)
            int trainSize = (int) (dataset.numExamples() * 0.8); // Calcule la taille du dataset et en prends 80% pour l'entrainement
            SplitTestAndTrain splits = dataset.splitTestAndTrain(trainSize);//Divise le dataset en deux parties avec les 80% calculés avant donc 80% entrainement et 20% test
            DataSet trainData = splits.getTrain(); // récupère les données pour l'entrainement
            DataSet testData = splits.getTest(); // récupère les données pour le test

            DataSetIterator trainIterator = new ListDataSetIterator<>(trainData.asList(), batchSize); // création d'un itérateur pour parcourir les données d'entrainement par lots de taille batchSize (128)
            DataSetIterator testIterator = new ListDataSetIterator<>(testData.asList(), batchSize); // création d'un itérateur pour parcourir les données de test par lots de taille batchSize (128)

            // Création et entraînement du modèle
            model = createModel(seed, learningRate); 
            
            // Entraînement du modèle
            System.out.println("Starting training...");
            trainModel(model, trainIterator, numEpochs);
           
            // Sauvegarde du modèle
            String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron4-model.zip";
            saveModel(model, modelPath);
            
            // Évaluation du modèle
            System.out.println("\nÉvaluation du modèle...");
            evaluateModel(modelPath, testIterator);
                    
            
        } catch (IOException | IllegalArgumentException e) {
            e.printStackTrace(); // gestion des erreurs 
        }
    }
    
    //Méthode pour entrainer le modèle
    private static void trainModel(MultiLayerNetwork model, DataSetIterator trainIterator, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) { // Effectue l'entrainement sur le nombre d'époques demandées
            trainIterator.reset(); // Initialise l'itérateur au début des données
            int batchNum = 0; // Compteur pour voir la progression des lots traités
            while (trainIterator.hasNext()) { // Traitement des données par lots
                DataSet batch = trainIterator.next(); // Prends le prochain lot de données
                model.fit(batch); // Entraine le modèle sur le lot
               
                if (batchNum % 10 == 0) { // Boucle qui permet d'afficher la progression de l'entrainement
                    System.out.printf("Epoch %d, Batch %d: Score = %.4f%n", 
                        epoch + 1, batchNum, model.score());
                }
                batchNum++; // indente le compteur
            }
            System.out.println("Completed epoch " + (epoch + 1));
        }
    }
    // Méthode pour évaluer le modèle 
    public static void evaluateModel(String modelPath, DataSetIterator testIterator) throws IOException {
         MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(modelPath)); // Charge le modèle à évaluer
         System.out.println("Modèle chargé depuis: " + modelPath);
         
         // Initialisations des variables
         RegressionEvaluation eval = new RegressionEvaluation(); // crée "eval" qui servira à évaluer les performances du modèle avec la bibliothèque DL4J
         int totalPredictions = 0; // création d'un compteur du nombre de prédiction
         int correctPredictions = 0; // création d'un compteur du nombre de prédiction correcte
         double mseSum = 0.0; // compteur correspondant à la somme des MSE calculées
         double threshold = 0.5; // seuil de classification
         
         //Boucle d'évaluation
         testIterator.reset(); // initialise l'itérateur de test au début des données
         while (testIterator.hasNext()) {
             DataSet batch = testIterator.next(); // prend le prochain lot de données
             INDArray features = batch.getFeatures(); // Données d'entrées que le modèle utilise pour faire les prédictions (la situation 64 valeurs) 
             INDArray labels = batch.getLabels(); // Valeurs réelles que l'on veut c'est à dire la 65eme valeur du dataset (la probabilité)
             INDArray predictions = model.output(features); // Prédictions du modèle
             eval.eval(labels, predictions); // calcule la correlation et MAE à l'aide de la librairie DL4J entre la probabilité du modèle (prédiction) et la vraie probabilité (celle du dataset "labels")
             
             // calcule des métriques 
             for (int i = 0; i < predictions.length(); i++) {
                 totalPredictions++; // indente le compteur des prédictions
                 double predicted = predictions.getDouble(i); // récupère la prédiction faite par le modèle pour la situation donnée
                 double actual = labels.getDouble(i); // récupère la valeur réelle pour la situation donnée
                 
                 //vérifie si la prédiction est correcte selon le seuil 0.5
                 if ((predicted >= threshold && actual >= threshold) ||
                     (predicted < threshold && actual < threshold)) {
                     correctPredictions++; // indente le compteur des prédictions correctes
                 }

                 mseSum += Math.pow(predicted - actual, 2); // calcule de l'erreur quadratique et l'ajoute à la somme des MSE
             }
         }

         double accuracy = (double) correctPredictions / totalPredictions * 100; // Précision du modèle
         double mse = mseSum / totalPredictions; // Erreur quadratique moyenne
         double rmse = Math.sqrt(mse); // Racine de l'erreur quadratique moyenne
         
         // Affiche les résultats
         System.out.println("\nRésultats de l'évaluation du modèle Perceptron :");
         System.out.println("----------------------------------------");
         System.out.printf("Précision de classification : %.2f%% (%d/%d)%n", 
                 accuracy, correctPredictions, totalPredictions);
         System.out.printf("Erreur quadratique moyenne (MSE) : %.4f%n", mse);
         System.out.printf("RMSE : %.4f%n", rmse);
         System.out.printf("Coefficient de corrélation : %.4f%n", eval.pearsonCorrelation(0));
         System.out.printf("MAE : %.4f%n", eval.averageMeanAbsoluteError());
     }



     

    // Fonction de création du dataset à partir d'un fichier CSV
    public static DataSet createDataset(String csvFilePath) throws IOException {
        List<INDArray> inputList = new ArrayList<>(); // création d'une liste INDArray (DL4J) correspondant aux entrées 
        List<INDArray> outputList = new ArrayList<>(); // Liste INdArray correspondant aux sorties
        
        //Lecture du fichier Csv ligne par ligne
        try (BufferedReader br = new BufferedReader(new FileReader(csvFilePath))) { 
            String line;
            int lineNumber = 0; //compteur du nombre de ligne
            while ((line = br.readLine()) != null) {
                lineNumber++; //indente le compteur du nombre de ligne 
                String[] values = line.split(","); // divise la ligne du csv pour récupérer les valeurs séparées par des virgules et donne un tableau de chaine qui contient les valeurs extraites 
                if (values.length != 65) { // Verifie que la ligne a 65 colonnes (64 entrées + 1 sortie)
                    System.err.println("Line " + lineNumber + " has " + values.length + " columns: " + line); // message d'erreur si la ligne n'a pas 65 colonnes 
                    continue; // Sauter cette ligne
                }

                // Créer l'entrée INDArray avec 64 valeurs
                INDArray input = Nd4j.zeros(1, 64); // initialisation du vecteur d'entrée avec 64 zéros
                boolean validInput = true; // booléen pour vérifier si l'entrée est valide ou non
                for (int i = 0; i < 64; i++) {
                    try {
                        input.putScalar(i, Double.parseDouble(values[i])); // chaque valeur d'entrée du csv est convertie en double et mise dans le vecteur
                    } catch (NumberFormatException e) {
                        System.err.println("Invalid number at line " + lineNumber + " column " + i + ": " + values[i]);
                        validInput = false;
                        break;
                    }
                }
                if (!validInput) continue; // passe si l'entrée n'est pas valide

                // Créer la sortie INDArray avec 1 valeur
                INDArray output = Nd4j.zeros(1, 1);
                try {
                    output.putScalar(0, Double.parseDouble(values[64]));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid number at line " + lineNumber + " column 64: " + values[64]);
                    continue;
                }
                
                // Ajoute les entrées et sorties validées à leurs listes respectives
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
        
        // crée l'objet dataset
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
            .seed(seed) // reproduit les résultats
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)//optimisation
            .updater(new Nesterovs(learningRate, 0.9)) // mise à jours des poids
            .weightInit(WeightInit.XAVIER) //initialisation des poids 
            .list()
            //Première couche cachée
            .layer(new DenseLayer.Builder()
                .nIn(64) // Nombre d'entrées (64 cases du plateau Othello)
                .nOut(256) //256 neuronnes
                .activation(Activation.LEAKYRELU)
                .build())
            //Deuxième couche cachée
            .layer(new DenseLayer.Builder()
                .nOut(256) //256 neuronnes 
                .activation(Activation.LEAKYRELU) // utilisation de la focntion d'activation RELU
                .build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE) // Utilisation de la MSE pour la régression
                .nOut(1) // Une seule sortie pour la probabilité de victoire
                .activation(Activation.IDENTITY) // Activation identité pour la régression
                .build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf); // crée l'objet model avec la configuration déterminée avant
        model.init();// initialisation des paramètres du modèle
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }
    
    
    
    
   
    
    // Fonction d'enregistrement du modèle
    public static void saveModel(MultiLayerNetwork model, String modelPath) throws IOException {
        System.out.println("Saving the model...");
        model.save(new File(modelPath));
        System.out.println("Model saved to " + modelPath);
    }

    
   
    
   

    
}

