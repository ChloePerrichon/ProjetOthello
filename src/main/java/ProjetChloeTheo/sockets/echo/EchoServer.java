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

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

/**
 * from https://www.baeldung.com/a-guide-to-java-sockets voir aussi
 * https://introcs.cs.princeton.edu/java/84network/EchoServer.java.html
 * https://stackoverflow.com/questions/33012456/bidirectional-clientserver-communication-in-java
 *
 * @author francois
 */
public class EchoServer {

    /** todoDoc. */
    public static int PORT = 6666;

    /**
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        ServerSocket serverSocket;
        Socket clientSocket;
        PrintWriter out;
        BufferedReader in;

        serverSocket = new ServerSocket(PORT);
        clientSocket = serverSocket.accept();
        out = new PrintWriter(clientSocket.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
        String req = in.readLine();

        out.println("serveur received : " + req);
        out.append(req+"\n");
    }
}
