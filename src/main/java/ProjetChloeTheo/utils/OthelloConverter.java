/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.utils;

/**
 *
 * @author toliveiragaspa01
 */
import ProjetChloeTheo.Apprentissage.CoupOthello;
import java.util.*;

public class OthelloConverter {

    public static int[][] convertirCoups(List<CoupOthello> coupsPossibles) {
        // Compter combien de coups "normaux" (pas "passe") il y a
        int count = 0;
        for (CoupOthello coup : coupsPossibles) {
            if (!coup.isPasse()) {
                count++;
            }
        }
        
        // Créer le tableau 2D avec une taille correspondant aux coups valides
        int[][] tableau = new int[count][2];
        
        // Remplir le tableau avec les coordonnées des coups valides
        int index = 0; //mettre le premier coup dans première colonne car la procédure chooseBestmove lit les coups en colonnes
        for (CoupOthello coup : coupsPossibles) {
            if (!coup.isPasse()) {
                tableau[index][0] = coup.getLig();  // Ligne
                tableau[index][1] = coup.getCol();  // Colonne
                index++;
            }
        }
        
        return tableau;
    }

    public static void main(String[] args) {
        // Exemple d'utilisation
        List<CoupOthello> coupsPossibles = new ArrayList<>();
        coupsPossibles.add(CoupOthello.coupNormal(3, 4));
        coupsPossibles.add(CoupOthello.coupNormal(5, 6));
        coupsPossibles.add(CoupOthello.coupPasse());
        coupsPossibles.add(CoupOthello.coupNormal(7, 8));

        int[][] tableau = convertirCoups(coupsPossibles);

        // Affichage du tableau 2D
        for (int i = 0; i < tableau.length; i++) {
            System.out.println("Coup " + (i + 1) + ": (" + tableau[i][0] + ", " + tableau[i][1] + ")");
        }
    }
}