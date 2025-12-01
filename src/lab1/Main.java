package lab1;

import mpi.*;
public class Main {
    public static void main(String[] args) {
        MPI.Init(args);

        int rank = MPI.COMM_WORLD.Rank();
        int size = MPI.COMM_WORLD.Size();
        int tag = 0;
        int[] message = new int[1];
        message[0] = rank;

        if (rank % 2 == 0) {
            if (rank + 1 != size) {
                System.out.println("Process " + rank + " sending " + message[0] + " to process" + (rank + 1));
                MPI.COMM_WORLD.Send(message, 0, 1, MPI.INT, rank + 1, tag);
            }
        } else {
            if (rank > 0) {
                MPI.COMM_WORLD.Recv(message, 0, 1, MPI.INT, rank - 1, tag);
                System.out.println("Process " + rank + " recieved " + message[0] + " from process" + (rank - 1));
            }
        }

        MPI.Finalize();
    }
}