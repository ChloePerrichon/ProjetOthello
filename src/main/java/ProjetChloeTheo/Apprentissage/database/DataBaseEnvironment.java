package ProjetChloeTheo.Apprentissage.database;

import java.io.*;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class DataBaseEnvironment {
    // Attributs nécessaires pour la connexion à la base de données
    private static final String URL = "jdbc:mysql://92.222.25.165:3306/m3_toliveiragaspar01"; //potentiellement rajouté : ?useSSL=false&serverTimeZone=UTC*
    private static final String USERNAME = "m3_toliveiragaspar01";
    private static final String PASSWORD = "24796a2c";
    private static Connection connection;

    // Connexion à la base de données
    public static Connection connect() {
        try {
            if (connection == null || connection.isClosed()) {
                connection = DriverManager.getConnection(URL, USERNAME, PASSWORD);
                System.out.println("Connexion etablie");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return connection;
    }

    // Fermer la connexion
    public static void close() {
        try {
            if (connection != null && !connection.isClosed()) {
                connection.close();
                System.out.println("Connexion close");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Méthode pour importer un fichier CSV en tant qu'objet dans la table "ressources_CSV".
     * @param cheminFichier Le chemin absolu du fichier CSV à importer.
     * @param nomTable Le nom de la table dans laquelle insérer le fichier (ex : "ressources_CSV").
     */
    public static void exporterCSVversDatabase(String cheminFichier, String nomTable) {
        String requete = "INSERT INTO " + nomTable + " (nom_fichier, nom_contenu_du_fichier) VALUES (?, ?)";
        try (PreparedStatement stmt = connection.prepareStatement(requete);
             FileInputStream fis = new FileInputStream(cheminFichier)) {

            // Obtenez le fichier et ses informations
            File fichier = new File(cheminFichier);
            stmt.setString(1, fichier.getName()); // Nom du fichier
            stmt.setBinaryStream(2, fis, (int) fichier.length()); // Contenu binaire du fichier

            // Exécutez la requête
            stmt.executeUpdate();
            System.out.println("Fichier " + fichier.getName() + " exporté dans la database ");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Méthode pour importer tous les fichiers CSV d'un dossier dans la table "ressources_CSV".
     * @param cheminDossier Le chemin absolu du dossier contenant les fichiers CSV.
     * @param nomTable Le nom de la table dans laquelle insérer les fichiers.
     */
    public static void exporterTousLesCSVduDossierVersDB(String cheminDossier, String nomTable) {
        File dossier = new File(cheminDossier);

        // Vérifiez si le chemin est un dossier valide
        if (dossier.isDirectory()) {
            for (File fichierCSV : dossier.listFiles()) {
                if (fichierCSV.isFile() && fichierCSV.getName().endsWith(".csv")) {
                    exporterCSVversDatabase(fichierCSV.getAbsolutePath(), nomTable);
                }
            }
        } else {
            System.out.println("Le chemin du dossier spécifié n'est pas un dossier valide.");
        }
    }

    /**
     * Méthode pour récupérer un fichier CSV depuis la table "ressources_CSV" et
     * le sauvegarder localement.
     *
     * @param id L'identifiant du fichier dans la table.
     * @param cheminDossierCible Le chemin du dossier où sauvegarder le fichier
     * récupéré.
     */
    public static void recupererCSVDepuisDatabase(int id, String cheminDossierCible) {
        String requete = "SELECT nom_fichier, contenu_fichier FROM ressources_CSV WHERE id = ?";
        try (PreparedStatement stmt = connection.prepareStatement(requete)) {
            stmt.setInt(1, id);

            try (ResultSet rs = stmt.executeQuery()) {
                if (rs.next()) {
                    String nomFichier = rs.getString("nom_fichier");
                    Blob blob = rs.getBlob("contenu_fichier");

                    // Écriture du fichier récupéré dans le dossier de sortie
                    try (InputStream is = blob.getBinaryStream(); FileOutputStream fos = new FileOutputStream(cheminDossierCible + File.separator + nomFichier)) {

                        byte[] buffer = new byte[1024];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }

                        System.out.println("Fichier importé depuis la database dans le dossier cible: " + nomFichier);
                    }
                } else {
                    System.out.println("Aucun fichier trouvé avec l'ID spécifié.");
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void exporterZIPversDatabase(String cheminFichier, String nomTable) {
        String requete = "INSERT INTO " + nomTable + " (nom_fichier, nom_contenu_du_fichier) VALUES (?, ?)";
        try (PreparedStatement stmt = connection.prepareStatement(requete); FileInputStream fis = new FileInputStream(cheminFichier)) {

            // Obtenez le fichier et ses informations
            File fichier = new File(cheminFichier);
            stmt.setString(1, fichier.getName()); // Nom du fichier
            stmt.setBinaryStream(2, fis, (int) fichier.length()); // Contenu binaire du fichier ZIP

            // Exécutez la requête
            stmt.executeUpdate();
            System.out.println("Fichier ZIP " + fichier.getName() + " exporté dans la base de données.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void exporterTousLesZIPduDossierVersDB(String cheminDossier, String nomTable) {
        File dossier = new File(cheminDossier);

        // Vérifiez si le chemin est un dossier valide
        if (dossier.isDirectory()) {
            for (File fichierZIP : dossier.listFiles()) {
                if (fichierZIP.isFile() && fichierZIP.getName().endsWith(".zip")) {
                    exporterZIPversDatabase(fichierZIP.getAbsolutePath(), nomTable);
                }
            }
        } else {
            System.out.println("Le chemin du dossier spécifié n'est pas un dossier valide.");
        }
    }

    public static void recupererZIPDepuisDatabase(int id, String cheminDossierCible) {
    String requete = "SELECT nom_fichier, nom_contenu_du_fichier FROM Model WHERE id = ?";
    try (PreparedStatement stmt = connection.prepareStatement(requete)) {
        stmt.setInt(1, id);

        try (ResultSet rs = stmt.executeQuery()) {
            if (rs.next()) {
                String nomFichier = rs.getString("nom_fichier");
                Blob blob = rs.getBlob("nom_contenu_du_fichier");

                // Écriture du fichier récupéré dans le dossier de sortie
                try (InputStream is = blob.getBinaryStream();
                     FileOutputStream fos = new FileOutputStream(cheminDossierCible + File.separator + nomFichier)) {

                    byte[] buffer = new byte[1024];
                    int bytesRead;
                    while ((bytesRead = is.read(buffer)) != -1) {
                        fos.write(buffer, 0, bytesRead);
                    }

                    System.out.println("Fichier ZIP importé depuis la base de données dans le dossier cible : " + nomFichier);
                }
            } else {
                System.out.println("Aucun fichier trouvé avec l'ID spécifié.");
            }
        }
    } catch (Exception e) {
        e.printStackTrace();
    }
}

    
    /**
     * Méthode pour créer une table nommée `ressources_CSV` dans la base de
     * données. La table aura les colonnes suivantes : - id : Identifiant unique
     * (clé primaire), généré automatiquement. - nom_fichier : Nom du fichier
     * (VARCHAR de 255 caractères, non nul). - nom_contenu_du_fichier : Contenu binaire du
     * fichier (LONGBLOB, non nul). - uploaded_at : Date et heure d'importation
     * du fichier (TIMESTAMP, valeur par défaut à l'heure actuelle).
     *
     * @param nomTable Le nom de la table à créer.
     * return Un entier représentant le statut de la création : 0 (succès), -1
     * (échec).
     */
    public static void CreerTableDeCSV(String nomTable) {
        // Définir la structure SQL de la table à créer
        String structureTable = "id INT AUTO_INCREMENT PRIMARY KEY, "
                + "nom_fichier VARCHAR(255) NOT NULL, "
                + "nom_contenu_du_fichier LONGBLOB NOT NULL, "
                + "uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP";  //un type BLOB permet de stocker des fichiers binaires, par exemple les fichiers binaires issus d'un CSV

        int creationtable = createTable(nomTable, structureTable);

        // Vérifier si la table a été créée avec succès
        if (creationtable == 0) {
            System.out.println("La table '" + nomTable + "' a été créée avec succès.");
        } else {
            System.out.println("Une erreur est survenue lors de la création de la table.");
        }
    }
    
    
   
    public static void CreerTableModeleIA(String nomTable) {
    // Définir la structure SQL de la table à créer
    String structureTable = "id INT AUTO_INCREMENT PRIMARY KEY, "
            + "nom_fichier VARCHAR(255) NOT NULL, "
            + "nom_contenu_du_fichier LONGBLOB NOT NULL, "  // Colonne pour stocker le fichier ZIP sous forme binaire
            + "uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP";  // Date et heure d'importation

    // Créer la table avec la structure spécifiée
    int creationtable = createTable(nomTable, structureTable);

    // Vérifier si la table a été créée avec succès
    if (creationtable == 0) {
        System.out.println("La table '" + nomTable + "' a été créée avec succès.");
    } else {
        System.out.println("Une erreur est survenue lors de la création de la table.");
    }
}

    // Créer une table dans la base de données
    public static int createTable(String tableName, String tableStructure) {
        // Construire la requête SQL pour créer la table
        String query = "CREATE TABLE " + tableName + " (" + tableStructure + ")";
        return executeUpdateQuery(query);
    }
    
        // Exécuter une requête INSERT, UPDATE, DELETE
    public static int executeUpdateQuery(String query) {
        // Exécuter la requête SQL
        try (Statement stmt = connection.createStatement()) {
            stmt.executeUpdate(query);
            return 0; // Succès
        } catch (SQLException e) {
            e.printStackTrace();
            return -1; // Échec
        }
    }
    
    public static void renommerTable(String ancienNomTable, String nouveauNomTable) {
    // Construire la requête SQL pour renommer la table
    String requete = "RENAME TABLE " + ancienNomTable + " TO " + nouveauNomTable;
    
    try (Statement stmt = connection.createStatement()) {
        // Exécuter la requête SQL
        stmt.executeUpdate(requete);
        System.out.println("La table a été renommée de '" + ancienNomTable + "' à '" + nouveauNomTable + "'.");
    } catch (SQLException e) {
        System.out.println("Une erreur est survenue lors du renommage de la table.");
        e.printStackTrace();
    }
}


    
    
    
    
    
    
    // Exécuter une requête SELECT (récupérer des données)
    public static List<List<Object>> executeSelectQuery(String query) {
        List<List<Object>> results = new ArrayList<>();
        try (Statement stmt = connect().createStatement()) {
            ResultSet rs = stmt.executeQuery(query);

            // Récupérer les métadonnées pour connaître les colonnes
            ResultSetMetaData metaData = rs.getMetaData();
            int columnCount = metaData.getColumnCount();

            // Parcourir les résultats et les ajouter à la liste
            while (rs.next()) {
                List<Object> row = new ArrayList<>();
                for (int i = 1; i <= columnCount; i++) {
                    row.add(rs.getObject(i));
                }
                results.add(row);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return results;
    }

    // Préparer une requête INSERT avec des paramètres
    public static int executePreparedUpdate(String query, Object[] parameters) {
        int rowsAffected = 0;
        try (PreparedStatement stmt = connect().prepareStatement(query)) {
            // Paramétrer la requête
            for (int i = 0; i < parameters.length; i++) {
                stmt.setObject(i + 1, parameters[i]);
            }
            rowsAffected = stmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return rowsAffected;
    }

    // Insérer une ligne dans une table
    public static int insertData(String table, Object[] data) {
        String query = "INSERT INTO " + table + " VALUES (" + String.join(",", "?" + "?" .repeat(data.length - 1)) + ")";
        return executePreparedUpdate(query, data);
    }

    // Mettre à jour une ligne dans une table
    
    public static int updateData(String table, String condition, Object[] data) {
        String query = "UPDATE " + table + " SET " + String.join(", ", (CharSequence[]) data) + " WHERE " + condition;
        return executePreparedUpdate(query, data);
    }

    // Supprimer une ligne dans une table
    public static int deleteData(String table, String condition) {
        String query = "DELETE FROM " + table + " WHERE " + condition;
        return executeUpdateQuery(query);
    }

    // Vérifier si une table existe
    public static boolean checkTableExists(String tableName) {
        try (ResultSet rs = connect().getMetaData().getTables(null, null, tableName, null)) {
            return rs.next();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return false;
    }

    // Exemple pour insérer plusieurs lignes à partir de données
    public static void insertBulkData(String tableName, List<Object[]> data) {
        try {
            String query = "INSERT INTO " + tableName + " VALUES (" + String.join(",", "?" .repeat(data.get(0).length)) + ")";
            try (PreparedStatement stmt = connect().prepareStatement(query)) {
                for (Object[] row : data) {
                    for (int i = 0; i < row.length; i++) {
                        stmt.setObject(i + 1, row[i]);
                    }
                    stmt.addBatch();
                }
                stmt.executeBatch();
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    /*
    // Exemple pour exporter des données vers un fichier CSV (en utilisant OpenCSV)
    public static void exportToCSV(String query, String filePath) {
        try (CSVWriter writer = new CSVWriter(new FileWriter(filePath))) {
            List<List<Object>> data = executeSelectQuery(query);
            for (List<Object> row : data) {
                writer.writeNext(row.stream().map(Object::toString).toArray(String[]::new));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    */
}
