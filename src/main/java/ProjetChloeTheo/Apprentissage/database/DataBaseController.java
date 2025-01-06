/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package ProjetChloeTheo.Apprentissage.database;

import java.util.List;

/**
 *
 * @author toliveiragaspa01
 */
public class DataBaseController {
    
    public static void main(String[] args) {
        // Connexion à la base de données
        DataBaseEnvironment.connect();
        
        // Exemple 1 : Insertion de données dans la table "utilisateurs"
        Object[] userData = {"John", "john.doe@example.com"};
        int rowsInserted = DataBaseEnvironment.insertData("utilisateurs", userData);
        System.out.println(rowsInserted + " ligne(s) insérée(s) dans la table 'utilisateurs'.");

        // Exemple 2 : Mise à jour des données dans la table "utilisateurs"
        Object[] updatedUserData = {"John Updated", "john.updated@example.com"};
        //int rowsUpdated = DataBaseEnvironment.updateData("utilisateurs", "id = 1", updatedUserData);
        //System.out.println(rowsUpdated + " ligne(s) mise(s) à jour dans la table 'utilisateurs'.");

        // Exemple 3 : Récupération des données de la table "utilisateurs"
        String selectQuery = "SELECT * FROM utilisateurs";
        List<List<Object>> users = DataBaseEnvironment.executeSelectQuery(selectQuery);
        System.out.println("Données récupérées :");
        for (List<Object> row : users) {
            System.out.println("Nom: " + row.get(0) + ", Email: " + row.get(1));
        }

        // Exemple 4 : Suppression d'un utilisateur dans la table "utilisateurs"
        int rowsDeleted = DataBaseEnvironment.deleteData("utilisateurs", "id = 1");
        System.out.println(rowsDeleted + " ligne(s) supprimée(s) dans la table 'utilisateurs'.");

        // Exemple 5 : Exportation des données de la table "utilisateurs" vers un fichier CSV
        String exportQuery = "SELECT * FROM utilisateurs";
        //DataBaseEnvironment.exportToCSV(exportQuery, "utilisateurs.csv");
        System.out.println("Exportation des données vers 'utilisateurs.csv' terminée.");

        // Fermer la connexion à la base de données
        DataBaseEnvironment.close();
    }
}
