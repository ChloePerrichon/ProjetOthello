/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage;

/**
 *
 * @author chloe
 */

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ValidationDataGenerator {
    
    static class ValidationPosition {
        double[] board;
        double evaluation;
        String description;
        
        public ValidationPosition(double[] board, double evaluation, String description) {
            this.board = board;
            this.evaluation = evaluation;
            this.description = description;
        }
    }
    
    public static void generateValidationData(String filePath) {
        try (FileWriter writer = new FileWriter(filePath)) {
            List<ValidationPosition> positions = createValidationPositions();
            
            for (ValidationPosition pos : positions) {
                // Écrire chaque position dans le fichier CSV
                StringBuilder line = new StringBuilder();
                
                // Ajouter les valeurs du plateau
                for (double value : pos.board) {
                    line.append(value).append(",");
                }
                
                // Ajouter l'évaluation
                line.append(pos.evaluation).append("\n");
                
                writer.write(line.toString());
                System.out.println("Position ajoutée: " + pos.description);
            }
            
            System.out.println("Fichier de validation créé avec succès: " + filePath);
            
        } catch (IOException e) {
            System.err.println("Erreur lors de la création du fichier de validation: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static List<ValidationPosition> createValidationPositions() {
        List<ValidationPosition> positions = new ArrayList<>();
        
        // Position initiale
        double[] initPosition = new double[64];
        initPosition[27] = 1;  // Blanc
        initPosition[28] = -1; // Noir
        initPosition[35] = -1; // Noir
        initPosition[36] = 1;  // Blanc
        positions.add(new ValidationPosition(initPosition, 0.5, "Position initiale"));
        
        // Position favorable aux noirs
        double[] blackAdvantage = new double[]{
            -1, -1, -1, -1, -1, -1, -1, -1,  // Première ligne
            -1,  0,  0,  0,  0,  0,  0, -1,
             0,  0,  1,  1, -1,  0,  0,  0,
             0,  0,  1, -1, -1,  1,  0,  0,
             0,  0, -1, -1,  1,  1,  0,  0,
             0,  0,  0, -1,  1,  1,  0,  0,
            -1,  0,  0,  0,  0,  0,  0, -1,
            -1, -1, -1, -1, -1, -1, -1, -1   // Dernière ligne
        };
        positions.add(new ValidationPosition(blackAdvantage, 0.8, "Avantage noir"));
        
        // Position favorable aux blancs (miroir de la position précédente)
        double[] whiteAdvantage = new double[64];
        for (int i = 0; i < 64; i++) {
            whiteAdvantage[i] = blackAdvantage[i] != 0 ? -blackAdvantage[i] : 0;
        }
        positions.add(new ValidationPosition(whiteAdvantage, 0.2, "Avantage blanc"));
        
        // Position de milieu de partie équilibrée
        double[] midGameBalanced = new double[]{
             0,  0, -1,  1,  1, -1,  0,  0,
             0, -1, -1,  1,  1, -1, -1,  0,
            -1, -1, -1,  1,  1, -1, -1, -1,
             1,  1,  1, -1, -1,  1,  1,  1,
             1,  1,  1, -1, -1,  1,  1,  1,
            -1, -1, -1,  1,  1, -1, -1, -1,
             0, -1, -1,  1,  1, -1, -1,  0,
             0,  0, -1,  1,  1, -1,  0,  0
        };
        positions.add(new ValidationPosition(midGameBalanced, 0.5, "Milieu équilibré"));
        
        // Position presque finale avec avantage noir
        double[] endGameBlack = new double[]{
            -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1,  1,  1, -1, -1, -1, -1,
            -1, -1,  1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1,  1,  1, -1, -1,
            -1, -1, -1,  1,  1,  1, -1, -1,
            -1, -1, -1, -1, -1, -1,  1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1
        };
        positions.add(new ValidationPosition(endGameBlack, 0.9, "Fin de partie noir"));
        
        return positions;
    }
    
    public static void main(String[] args) {
        String filePath = "src\\main\\java\\ProjetChloeTheo\\Ressources\\CsvAvecEntrainement\\validation.csv";
        generateValidationData(filePath);
    }
}