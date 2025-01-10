package ProjetChloeTheo.Apprentissage.database;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class DataBaseEnvironment {
    // Attributs nécessaires pour la connexion à la base de données
    private static final String URL = "jdbc:mysql://92.222.25.165:3306/m3_toliveiragaspar01"; //potentiellement rajouté : ?useSSL=false&serverTimeZone=UTC*
    private static final String USERNAME = "m3_toliveiragaspar01";
    private static final String PASSWORD = "";
    private static Connection connection;

    // Connexion à la base de données
    public static Connection connect() {
        try {
            if (connection == null || connection.isClosed()) {
                connection = DriverManager.getConnection(URL, USERNAME, PASSWORD);
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
            }
        } catch (SQLException e) {
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

    // Exécuter une requête INSERT, UPDATE, DELETE
    public static int executeUpdateQuery(String query) {
        int rowsAffected = 0;
        try (Statement stmt = connect().createStatement()) {
            rowsAffected = stmt.executeUpdate(query);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return rowsAffected;
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
    
    //public static int updateData(String table, String condition, Object[] data) {
    //    String query = "UPDATE " + table + " SET " + String.join(", ", data) + " WHERE " + condition;
    //    return executePreparedUpdate(query, data);
    //}

    // Supprimer une ligne dans une table
    public static int deleteData(String table, String condition) {
        String query = "DELETE FROM " + table + " WHERE " + condition;
        return executeUpdateQuery(query);
    }

    // Créer une table dans la base de données
    public static int createTable(String tableName, String tableStructure) {
        String query = "CREATE TABLE " + tableName + " (" + tableStructure + ")";
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
