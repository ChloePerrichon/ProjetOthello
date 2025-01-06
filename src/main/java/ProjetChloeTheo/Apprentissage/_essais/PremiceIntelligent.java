/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage._essais;

import ProjetChloeTheo.Apprentissage.config_Othello.CoupOthello;
import ProjetChloeTheo.Apprentissage.config_Othello.JeuOthello;
import ProjetChloeTheo.Apprentissage.config_Othello.Joueur;
import ProjetChloeTheo.Apprentissage.config_Othello.SituationOthello;
import ProjetChloeTheo.configuration_jeu_othello.Damier;
import ProjetChloeTheo.utils.OthelloConverter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author toliveiragaspa01
 */
public class PremiceIntelligent {
    
    // Classe pour stocker une situation unique et ses résultats
    static class SituationLigneATester {
        int[] board; // Représentation du damier 8x8 (64 cases)
        double probability; // Probabilité de victoire pour cette situation

        public SituationLigneATester(int[] board, double probability) {
            this.board = board;
            this.probability = probability;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof SituationLigneATester)) return false;
            SituationLigneATester other = (SituationLigneATester) obj;
            return Arrays.equals(this.board, other.board);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(board);
        }
    }

    // Fonction pour lire un fichier et charger les probabilités
    public static Map<SituationLigneATester, Double> loadProbabilitiesofOnefile(String filePath) {
        Map<SituationLigneATester, Double> probabilities = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                int[] board = new int[64];
                for (int i = 0; i < 64; i++) {
                    board[i] = Integer.parseInt(values[i]);
                }
                double result;
                result = Double.parseDouble(values[64]); // Valeur de la 65ème colonne

                SituationLigneATester situation = new SituationLigneATester(board, result);
                probabilities.put(situation, result); // Charger directement les probabilités
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return probabilities;
    }

    // Fonction pour trouver les probabilités de la situation actuelle dans plusieurs fichiers
    public static double getProbabilityInAllFilesForCurrentSituation(int[] currentBoard, List<String> filePaths) {
        for (String filePath : filePaths) {
            Map<SituationLigneATester, Double> probabilities = loadProbabilitiesofOnefile(filePath);
            for (SituationLigneATester situation : probabilities.keySet()) {
                if (Arrays.equals(situation.board, currentBoard)) {
                    return probabilities.get(situation); // Retourner la probabilité si trouvée
                }
            }
        }
        return 0.5; // Retourner une valeur moyenne de proba si situation inédite
    }

    // Fonction pour choisir le meilleur coup à jouer
    public static int[] chooseBestMove(int[][] possibleMoves, List<String> filePaths) {
        double maxProbability = -1.0;   //valeur initiale qui sera incrémentée si meilleur proba trouvée jusqu'à être maximale
        int[] bestMove = null;          //initial si aucun meilleur coup trouvé

        for (int[] move : possibleMoves) { // Pour chaque coup possible (affiché sur une ligne du tableau possibleMoves)
            double probability = getProbabilityInAllFilesForCurrentSituation(move, filePaths);
            if (probability > maxProbability) {
                maxProbability = probability;
                bestMove = move;
            }
        }

        return bestMove; // Retourner le meilleur coup (ou null si aucun trouvé)
    }
    
    public static String ConverttoString(int [] p) {
        String retourner = "[" + (char) ('A' + p[0]) + "," + (p[1]+1) + "]";
        return retourner;
    }

    // Méthode pour tester l'oracle intelligent
    public static void main(String[] args) {
        // Liste des fichiers contenant les données
        List<String> filePaths = Arrays.asList("C:\\tmp\\blancs8000.csv","C:\\tmp\\noirs8000.csv",
                "C:\\tmp\\blancs9000.csv","C:\\tmp\\noirs9000.csv");

        // Situation initale du jeu (par exemple)
        /* Espace de travail Chloe Theo
        JeuOthello jeu = new JeuOthello();
        try {
            File dir = new File("C:\\tmp\\fichiers_entrainement");
            creationPartie(new File(dir, "noirsEntraines" + nbr +".csv"), new File(dir, "blancsNuls"+nbr+".csv"),
                    new JeuOthello(), new OracleIntelligent(Joueur.NOIR), new OracleStupide(Joueur.BLANC),nbr, true, true, true,new Random());
        } catch (IOException ex){
            throw new Error(ex);
        }
        */
        //int[] currentBoard = new int[64];
        //Arrays.fill(currentBoard, 0); // Par exemple, toutes les cases vides

        //Initialisation du jeu avec situation initiale, et damier initial
        JeuOthello jeu = new JeuOthello();
        Damier board = new Damier();
        SituationOthello situation = new SituationOthello(board);
        
        var joueurs = List.of(Joueur.NOIR, Joueur.BLANC);
        int numJoueur = 0;      // 0 pour NOIR, 1 pour BLANC
        Joueur curJoueur = joueurs.get(numJoueur);
                
        List<CoupOthello> CoupsPossibles = jeu.coupsJouables(situation, curJoueur);
        
//        for (CoupOthello coup : CoupsPossibles){
//            System.out.println("CoupsPossibles est : ");
//            System.out.println(coup.toString());
//        }
        
        //Convert list en tableau de int 2D
        int[][] possibleMoves = OthelloConverter.convertirCoups(CoupsPossibles);
        //int[][] possibleMoves = ListToArray.listToArray(CoupsPossibles);
        
        for (int[] move : possibleMoves) {
            System.out.println("possibleMoves est : ");
            //System.out.println(Arrays.toString(move));
            System.out.println(ConverttoString(move));
        }
        
        // Trouver et jouer le meilleur coup
        int[] bestMove = chooseBestMove(possibleMoves, filePaths);
        if (bestMove != null) {
            System.out.println("Meilleur coup trouver : " + ConverttoString(bestMove));
        } else {
            System.out.println("Aucun coup gagnant trouvé dans les fichiers.");
        }
    }
}

