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

    public static void generateCSVOfSituations(         //méthode qui crée les lignes du CSV une par une
            Writer outJ1, Writer outJ2,
            JeuOthello jeu, Oracle j1, Oracle j2,
            int nbrParties,
            boolean includeRes, boolean includeNumCoup, boolean includeTotCoup,
            Random rand) throws IOException {
        
        int Quigagne[]= new int[2];
        
        for (int i = 0; i < nbrParties; i++) {
            ResumeResultat resj = jeu.partie(j1,ChoixCoup.ORACLE_MEILLEUR,    // LIGNE POUR MODIFIER LES CHOIX COUPS DES ORACLES
                    j2, ChoixCoup.ORACLE_PONDERE, false, false, rand,false);  //je joue la partie ici !!!
            // je rejoue la partie pour avoir les situations
            SituationOthello curSit = jeu.situationInitiale();
            Writer curOut = outJ1;
            double curRes;
            if (resj.getStatutFinal() == StatutSituation.NOIR_GAGNE) {
                curRes = 1;
                Quigagne[0]++;
            } else if (resj.getStatutFinal() == StatutSituation.BLANC_GAGNE) {
                curRes = 0;
                Quigagne[1]++;
            } else if (resj.getStatutFinal() == StatutSituation.MATCH_NUL) {
                curRes = 0.5;
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
        System.out.println("les noirs ont gagne "+Quigagne[0]+" fois");
        System.out.println("les blancs ont gagne "+Quigagne[1]+" fois");
    }

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
    
     public static void testAvecOthelloV2(int nbr, String modelPath,String modelPath1) {
        try {
            //File dir = new File("C:\\tmp");
            File dir = new File("src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement");
            JeuOthello jeu = new JeuOthello();
            Oracle j1 = new OraclePerceptron(Joueur.NOIR, modelPath);
            Oracle j2 = new OraclePerceptron(Joueur.BLANC, modelPath1);
            

            creationPartie(new File(dir, "noirs" + nbr + "OM-OP.csv"), new File(dir, "blancs" + nbr + "OM-OP.csv"),
                    jeu, j1, j2, nbr, true, false, false, new Random());
            
        } catch (IOException ex) {
            throw new Error(ex);
        }
    }
    
    public static void main(String[] args){
        //testAvecOthello(10000);
        // Chemin du modèle entraîné pour l'OracleIntelligent
        String modelPath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-mlp-model-AvecEntraienmentOPcontreOP-noir-5000.zip";
        String modelPath1 = "src\\main\\java\\ProjetChloeTheo\\Ressources\\Model\\othello-mlp-model-AvecEntraienmentOPcontreOP-noir-5000.zip";
        testAvecOthelloV2(5000, modelPath,modelPath1);
    }

}
