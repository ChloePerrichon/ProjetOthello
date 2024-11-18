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
package ProjetChloeTheo.sockets.echo;

import ProjetChloeTheo.utils.ConsoleFdB;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

/**
 *
 * @author francois
 */
public class EchoClient {

    /**
     *
     * @param args
     */
    public static void main(String[] args) {

        try {
            Socket clientSocket;
            PrintWriter out;
            BufferedReader in;
            
            clientSocket = new Socket("127.0.0.1", EchoServer.PORT);
            out = new PrintWriter(clientSocket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            String msg = ConsoleFdB.entreeString("requete : ");
            out.println(msg);
            String resp = in.readLine();
            System.out.println(resp);
            in.close();
            out.close();
            clientSocket.close();
        } catch (IOException ex) {
            throw new Error(ex);
        }

    }

}
