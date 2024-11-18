/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author cperrichon01
 */

import java.io.*;
import java.util.*;

public class CalculProbaTtesSituationsUnFichier {

    // Classe pour stocker une situation unique et ses résultats
    static class SituationLigne {
        int[] board; // Représentation du damier 8x8 (64 cases)
        int outcomes; // Nombre total de résultats pour cette situation
        double wins; // Nombre de victoires associées (score de 1 pour noirs)

        public SituationLigne(int[] board) {
            this.board = board;
            this.outcomes = 0;
            this.wins = 0.0;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof SituationLigne)) return false;
            SituationLigne other = (SituationLigne) obj;
            return Arrays.equals(this.board, other.board);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(board);
        }
    }

    public static Map<SituationLigne, Double> calculateWinningProbabilities(String filePath) {
        Map<SituationLigne, SituationLigne> situationMap = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                int[] board = new int[64];
                for (int i = 0; i < 64; i++) {
                    board[i] = Integer.parseInt(values[i]);
                }
                double result = Double.parseDouble(values[64]); // Valeur de la 65ème colonne

                SituationLigne situation = new SituationLigne(board);
                if (!situationMap.containsKey(situation)) {
                    situationMap.put(situation, situation);
                }

                SituationLigne existing = situationMap.get(situation);
                existing.outcomes++;
                existing.wins += result; // Somme des victoires (1 si noir, 0.5 si nul)
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Calcul des probabilités
        Map<SituationLigne, Double> probabilities = new HashMap<>();
        for (SituationLigne situation : situationMap.values()) {
            probabilities.put(situation, situation.wins / situation.outcomes);
        }
        return probabilities;
    }

    // Méthode pour tester
    public static void main(String[] args) {
        String filePath = "C:\\tmp\\noirs8000.csv"; // Remplacer par le chemin réel
        Map<SituationLigne, Double> probabilities = calculateWinningProbabilities(filePath);

        // Afficher les probabilités pour toutes les situations uniques
        for (Map.Entry<SituationLigne, Double> entry : probabilities.entrySet()) {
            System.out.println("Situation: " + Arrays.toString(entry.getKey().board));
            System.out.println("Probability of winning: " + entry.getValue());
        }
    }
}
