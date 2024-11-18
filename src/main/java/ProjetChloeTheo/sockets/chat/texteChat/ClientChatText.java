/*
Copyright 2000-2014 Francois de Bertrand de Beuvron

This file is part of CoursBeuvron.

CoursBeuvron is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CoursBeuvron is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CoursBeuvron.  If not, see <http://www.gnu.org/licenses/>.
 */
package ProjetChloeTheo.sockets.chat.texteChat;

import ProjetChloeTheo.utils.ConsoleFdB;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.Socket;
import java.nio.charset.Charset;

/**
 *
 * @author francois
 */
public class ClientChatText {

    /** todoDoc. */
    public static void start() {
        String adr = ConsoleFdB.entreeString("Adresse host : ");
        int port = ConsoleFdB.entreeInt("port : ");
        try (Socket sock = new Socket(adr, port)) {
            ClientListerner listen = new ClientListerner(sock);
            listen.start();
            try (Writer bout = new OutputStreamWriter(sock.getOutputStream(), Charset.forName("UTF8"))) {
                System.out.println();
                String nom = ConsoleFdB.entreeString("donnez votre nom : ");
                bout.append(nom+"\n");
                System.out.println("ready !! type text then <enter>");
                while (true) {
                    String mess = ConsoleFdB.entreeString("");
                    bout.append(mess+"\n");
                    bout.flush();
                }
            }

        } catch (IOException ex) {
            throw new Error(ex);
        }
    }

    /** todoDoc. */
    public static class ClientListerner extends Thread {

        private Socket sock;

        /**
         *
         * @param sock
         */
        public ClientListerner(Socket sock) {
            this.sock = sock;
        }

        /** todoDoc. */
        @Override
        public void run() {
            try (BufferedReader bin = new BufferedReader(new InputStreamReader(this.sock.getInputStream(), Charset.forName("UTF8")))) {
                String line;
                while ((line = bin.readLine()) != null) {
                    System.out.println(line);
                }
            } catch (IOException ex) {
                throw new Error(ex);
            }

        }
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        start();
    }

}
