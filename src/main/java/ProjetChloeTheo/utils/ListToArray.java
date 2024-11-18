/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.utils;

/**
 *
 * @author toliveiragaspa01
 */
import java.util.*;

public class ListToArray {
    
    public static void main(String[] args) {
        // Création d'une List<List<Integer>> pour l'exemple
        List<List<Integer>> list = new ArrayList<>();
        
        // Ajouter des sous-listes
        list.add(Arrays.asList(1, 2, 3));
        list.add(Arrays.asList(4, 5, 6));
        list.add(Arrays.asList(7, 8, 9));

        // Convertir la List<List<Integer>> en tableau int[][]
        int[][] array = listToArray(list);

        // Afficher le tableau 2D
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                System.out.print(array[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static int[][] listToArray(List<List<Integer>> list) {
        int rows = list.size();
        int cols = list.get(0).size();
        
        int[][] array = new int[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            List<Integer> row = list.get(i);
            for (int j = 0; j < cols; j++) {
                array[i][j] = row.get(j);
            }
        }
        
        return array;
    }
}
