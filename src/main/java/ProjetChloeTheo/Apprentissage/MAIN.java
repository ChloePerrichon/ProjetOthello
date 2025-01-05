/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Random;
import org.nd4j.linalg.dataset.DataSet;

/**
 *
 * @author francois
 */
public class MAIN {
    
    
    
    //Génère une ligne CSV pour une situation donnée
    
    private static void generateUneLigneCSVOfSituations(
            Writer curWriter,
            SituationOthello curSit, double res, int numCoup, int totCoup,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup) throws IOException {
        curWriter.append(curSit.toCSV());
        if (includeRes) {
            curWriter.append("," + res);
        }
        if (includeNumCoup) {
            curWriter.append("," + numCoup);
        }
        if (includeTotCoup) {
            curWriter.append("," + totCoup);
        }
        curWriter.append("\n");
    }
    
    
    
    // Génère le CSV pour les situations de jeu
    
    public static void generateCSVOfSituations(         
            Writer outJ1, Writer outJ2,
            JeuOthello jeu, Oracle j1, Oracle j2,
            int nbrParties,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup,
            Random rand) throws IOException {
        
        int Quigagne[]= new int[2]; //[0] pour noir, [1] pour blanc
        
        
        for (int i = 0; i < nbrParties; i++) {
            ResumeResultat resj = jeu.partie(j1,ChoixCoup.ORACLE_MEILLEUR,    // LIGNE POUR MODIFIER LES CHOIX COUPS DES ORACLES
                    j2, ChoixCoup.ORACLE_PONDERE, false, false, rand,false);  
            SituationOthello curSit = jeu.situationInitiale();
            Writer curOut = outJ1;
            double curRes;
            if (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) {
                curRes = 1;
                System.out.println("Partie " + (i + 1) + ": Les noirs ont gagné");
                Quigagne[0]++;
            } else if (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) {
                curRes = 0;
                System.out.println("Partie " + (i + 1) + ": Les blancs ont gagné");
                Quigagne[1]++;
            } else if (resj.getStatutFinal() == StatutSituation.MATCH_NUL) {
                curRes = 0.5;
                System.out.println("Partie " + (i + 1) + ": Match nul");
            } else {
                throw new Error("partie non finie");
            }
            int totCoups = resj.getCoupsJoues().size();
            int numCoup = 0;
            generateUneLigneCSVOfSituations(curOut, curSit, curRes, numCoup, totCoups, includeRes, includeNumCoup, includeTotCoup);  //générer la ligne initiale de la situation init
            Joueur curJoueur = Joueur.NOIR;
            for (CoupOthello curCoup : resj.getCoupsJoues()) {
                curSit = jeu.updateSituation(curSit, curJoueur, curCoup);
                if (curOut == outJ1) {
                    curOut = outJ2;
                } else {
                    curOut = outJ1;
                }
                curRes = 1 - curRes;
                numCoup++;
                generateUneLigneCSVOfSituations(curOut, curSit, curRes, numCoup, totCoups, includeRes, includeNumCoup, includeTotCoup); //générer nouvelle ligne à chaque coup
                curJoueur = curJoueur.adversaire();
            }
        }
        
        // Affichage des pourcentages de victoire
        int totalParties = Quigagne[0] + Quigagne[1];
        double pourcentageNoirs = (double) Quigagne[0] / totalParties * 100;
        double pourcentageBlancs = (double) Quigagne[1] / totalParties * 100;

        System.out.printf("Les noirs ont gagné %d fois (%.2f%%)\n", Quigagne[0], pourcentageNoirs);
        System.out.printf("Les blancs ont gagné %d fois (%.2f%%)\n", Quigagne[1], pourcentageBlancs);
        
    }
    
    // Crée le fichier csv pour une série de parties
    public static void creationPartie(
            File outJ1, File outJ2,
            JeuOthello jeu, Oracle j1, Oracle j2,
            int nbrParties,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup,
            Random rand) throws IOException {
        try (FileWriter wJ1 = new FileWriter(outJ1); FileWriter wJ2 = new FileWriter(outJ2)) {
            generateCSVOfSituations(wJ1, wJ2, jeu, j1, j2, nbrParties, includeRes, includeNumCoup, includeTotCoup,rand);
        }
    }
    
    // Test de base avec les oracles stupides
    public static void testAvecOthello(int nbr) {
        try {
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvSansEntrainement"); //attention je crois que sur l'ordi de chloe c'est temp
            //File dir = new File("C:\\temp");
            creationPartie(new File(dir, "noirs" + nbr +".csv"), new File(dir, "blancs"+nbr+".csv"),
                    new JeuOthello(), new OracleStupide(Joueur.NOIR), new OracleStupide(Joueur.BLANC),nbr, true, false, false,new Random());
        } catch (IOException ex){
            throw new Error(ex);
        }
    }
    
    //Test avec les oracles intelligents et création des fichiers csv normal et avec proba
     public static void testAvecOthelloV2(int nbr, String modelPath,String modelPath1) {
        try {
           
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
            
            // Création des parties et générations du csv normal
            JeuOthello jeu = new JeuOthello();            
            Oracle j1 = new OracleCNN(Joueur.NOIR, modelPath, true);
            Oracle j2 = new OraclePerceptron(Joueur.BLANC, modelPath1,true);
            
            System.out.println("Modèles chargés avec succès.");
            System.out.println("\nDébut des parties...");
            
            creationPartie(
                    new File(dir, "noirs" + nbr + "OMCNN-OPPER.csv"), 
                    new File(dir, "blancs" + nbr + "OMCNN-OPPER.csv"),
                    jeu, j1, j2, nbr, true, false, false, new Random());
            
            System.out.println("Création du fichier csv avec proba ...");
            // Création du fichier csv avec proba
            String inputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\noirs" + nbr + "OMCNN-OPPER.csv";
            String outputFile = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainementProba\\noirsProba" + nbr + "OMCNN-OPPER.csv";
            CsvAvecProba.createCsvProba(inputFile, outputFile);
            
            
        } catch (IOException ex) {
            throw new Error(ex);
        }
    }
    
        
    
    public static void main(String[] args){
        //testAvecOthello(10000);
        //CNNModelValidator.validateModel(modelPath);
        // Chemin du modèle entraîné pour l'OracleIntelligent
        String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-cnn-model.zip";
        String modelPath1 = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-perceptron-model-OMPER-OPPER.zip";
        //Lancement du test
        testAvecOthelloV2(8000, modelPath,modelPath1);
    }
    
    

}
